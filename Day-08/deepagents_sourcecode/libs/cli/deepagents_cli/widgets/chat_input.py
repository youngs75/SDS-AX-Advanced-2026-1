"""Textual 앱에서 사용하는 기본 채팅 입력 영역을 제공합니다.

이 모듈은 여러 줄 편집, 슬래시 명령 및 파일 완성, 기록 탐색, 모드 접두사 및 붙여넣은 미디어 자리 표시자를 조정합니다.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.content import Content
from textual.css.query import NoMatches
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Static, TextArea

from deepagents_cli import theme
from deepagents_cli.command_registry import SLASH_COMMANDS
from deepagents_cli.config import (
    MODE_DISPLAY_GLYPHS,
    MODE_PREFIXES,
    PREFIX_TO_MODE,
    is_ascii_mode,
)
from deepagents_cli.input import IMAGE_PLACEHOLDER_PATTERN, VIDEO_PLACEHOLDER_PATTERN
from deepagents_cli.widgets.autocomplete import (
    CompletionResult,
    FuzzyFileController,
    MultiCompletionManager,
    SlashCommandController,
)
from deepagents_cli.widgets.history import HistoryManager

logger = logging.getLogger(__name__)


def _default_history_path() -> Path:
    """기본 기록 파일 경로를 반환합니다.

    테스트가 이를 임시 경로에 패치할 수 있도록 함수로 추출되어 테스트 실행이 `~/.deepagents/history.jsonl`을 오염시키는 것을
    방지합니다.

    """
    return Path.home() / ".deepagents" / "history.jsonl"


_PASTE_BURST_CHAR_GAP_SECONDS = 0.03
"""입력을 페이스트 같은 버스트로 처리하는 문자 사이의 최대 시간입니다."""

_PASTE_BURST_FLUSH_DELAY_SECONDS = 0.08
"""버퍼링된 버스트 텍스트를 플러시하기 전 유휴 시간 초과입니다."""

_PASTE_BURST_START_CHARS = {"'", '"'}
"""삭제된 경로 페이로드를 시작할 수 있는 문자입니다."""

_BACKSLASH_ENTER_GAP_SECONDS = 0.15
"""`\\` 키와 다음 `enter` 키 사이의 최대 간격은 쌍을 터미널에서 방출되는 Shift+Enter 시퀀스로 처리합니다.

일부 터미널(예: VSCode의 내장 터미널)은 사용자가 Shift+Enter를 누를 때 문자 그대로 백슬래시와 Enter를 보냅니다.  터미널이 두 문자를
거의 동시에 내보내므로 간격이 넉넉합니다(150ms). 사람이 의도적으로 `\\`을 입력한 다음 Enter를 누르면 간격이 훨씬 더 커집니다.
"""

if TYPE_CHECKING:
    from textual import events
    from textual.app import ComposeResult
    from textual.events import Click
    from textual.timer import Timer

    from deepagents_cli.input import MediaTracker, ParsedPastedPathPayload


# ---------------------------------------------------------------------------
# Completion popup widgets and their event plumbing
# ---------------------------------------------------------------------------

class CompletionOption(Static):
    """자동 완성 팝업에서 클릭 가능한 완성 옵션입니다."""

    DEFAULT_CSS = """
    CompletionOption {
        height: 1;
        padding: 0 1;
    }

    CompletionOption:hover {
        background: $surface-lighten-1;
    }

    CompletionOption.completion-option-selected {
        background: $primary;
        color: $background;
        text-style: bold;
    }

    CompletionOption.completion-option-selected:hover {
        background: $primary-lighten-1;
    }
    """

    class Clicked(Message):
        """완료 옵션을 클릭하면 전송되는 메시지입니다."""

        def __init__(self, index: int) -> None:
            """클릭한 옵션 인덱스로 초기화합니다."""
            super().__init__()
            self.index = index

    def __init__(
        self,
        label: str,
        description: str,
        index: int,
        is_selected: bool = False,
        **kwargs: Any,
    ) -> None:
        """완료 옵션을 초기화합니다.

        Args:
            label: 기본 레이블 텍스트(예: 명령 이름 또는 파일 경로)
            description: 보조 설명 텍스트
            index: 제안 목록에 있는 이 옵션의 색인
            is_selected: 이 옵션이 현재 선택되어 있는지 여부
            **kwargs: 부모에 대한 추가 인수

        """
        super().__init__(**kwargs)
        self._label = label
        self._description = description
        self._index = index
        self._is_selected = is_selected

    def on_mount(self) -> None:
        """마운트 시 옵션 표시를 설정합니다."""
        self._update_display()

    def _update_display(self) -> None:
        """표시 텍스트와 스타일을 업데이트합니다."""
        display_label = self._label.removeprefix("/")
        if self._description:
            content = Content.from_markup(
                "[bold]$label[/bold]  [dim]$desc[/dim]",
                label=display_label,
                desc=self._description,
            )
        else:
            content = Content.from_markup("[bold]$label[/bold]", label=display_label)

        self.update(content)

        if self._is_selected:
            self.add_class("completion-option-selected")
        else:
            self.remove_class("completion-option-selected")

    def set_selected(self, *, selected: bool) -> None:
        """이 옵션의 선택된 상태를 업데이트합니다."""
        if self._is_selected != selected:
            self._is_selected = selected
            self._update_display()

    def set_content(
        self, label: str, description: str, index: int, *, is_selected: bool
    ) -> None:
        """레이블, 설명, 색인 및 선택 항목을 제자리에서 바꿉니다."""
        self._label = label
        self._description = description
        self._index = index
        self._is_selected = is_selected
        self._update_display()

    def on_click(self, event: Click) -> None:
        """이 옵션을 클릭하세요."""
        event.stop()
        self.post_message(self.Clicked(self._index))


class CompletionPopup(VerticalScroll):
    """클릭 가능한 옵션으로 완성 제안을 표시하는 팝업 위젯입니다."""

    DEFAULT_CSS = """
    CompletionPopup {
        display: none;
        height: auto;
        max-height: 12;
    }
    """

    class OptionClicked(Message):
        """완료 옵션을 클릭하면 전송되는 메시지입니다."""

        def __init__(self, index: int) -> None:
            """클릭한 옵션 인덱스로 초기화합니다."""
            super().__init__()
            self.index = index

    def __init__(self, **kwargs: Any) -> None:
        """완료 팝업을 초기화합니다."""
        super().__init__(**kwargs)
        self.can_focus = False
        self._options: list[CompletionOption] = []
        self._selected_index = 0
        self._pending_suggestions: list[tuple[str, str]] = []
        self._pending_selected: int = 0
        self._rebuild_generation: int = 0

    def update_suggestions(
        self, suggestions: list[tuple[str, str]], selected_index: int
    ) -> None:
        """새로운 제안으로 팝업을 업데이트하세요."""
        if not suggestions:
            self.hide()
            return

        self._selected_index = selected_index
        self._pending_suggestions = suggestions
        self._pending_selected = selected_index
        # Increment generation so stale callbacks from prior calls are skipped.
        self._rebuild_generation += 1
        gen = self._rebuild_generation
        # show() deferred to _rebuild_options to avoid a flash of stale content.
        self.call_after_refresh(lambda: self._rebuild_options(gen))

    async def _rebuild_options(self, generation: int) -> None:
        """보류 중인 제안에서 옵션 위젯을 다시 빌드합니다.

        팝업이 표시되는 동안 전체 분해/마운트 주기로 인한 깜박임을 방지하기 위해 가능한 경우 기존 DOM 노드를 재사용합니다.

        Args:
            generation: 발신자의 생성 카운터; 대체된 경우 건너뜁니다.

        """
        if generation != self._rebuild_generation:
            return

        suggestions = self._pending_suggestions
        selected_index = self._pending_selected

        if not suggestions:
            self.hide()
            return

        existing = len(self._options)
        needed = len(suggestions)

        # Update existing widgets in-place
        for i in range(min(existing, needed)):
            label, desc = suggestions[i]
            self._options[i].set_content(
                label, desc, i, is_selected=(i == selected_index)
            )

        # DOM mutations: trim extras / mount new widgets
        try:
            if existing > needed:
                for option in self._options[needed:]:
                    await option.remove()
                del self._options[needed:]

            if needed > existing:
                new_widgets: list[CompletionOption] = []
                for idx in range(existing, needed):
                    label, desc = suggestions[idx]
                    option = CompletionOption(
                        label=label,
                        description=desc,
                        index=idx,
                        is_selected=(idx == selected_index),
                    )
                    new_widgets.append(option)
                self._options.extend(new_widgets)
                await self.mount(*new_widgets)
        except Exception:
            logger.exception("Failed to rebuild completion popup; hiding to recover")
            self._options = []
            with contextlib.suppress(Exception):
                await self.remove_children()
            self.hide()
            return

        self.show()

        if 0 <= selected_index < len(self._options):
            self._options[selected_index].scroll_visible()

    def update_selection(self, selected_index: int) -> None:
        """목록을 다시 작성하지 않고 어떤 옵션이 선택되었는지 업데이트합니다."""
        # Keep pending state in sync so an in-flight _rebuild_options uses
        # the latest selection.
        self._pending_selected = selected_index

        if self._selected_index == selected_index:
            return

        # Deselect previous
        if 0 <= self._selected_index < len(self._options):
            self._options[self._selected_index].set_selected(selected=False)

        # Select new
        self._selected_index = selected_index
        if 0 <= selected_index < len(self._options):
            self._options[selected_index].set_selected(selected=True)
            self._options[selected_index].scroll_visible()

    def on_completion_option_clicked(self, event: CompletionOption.Clicked) -> None:
        """완료 옵션을 클릭하여 처리합니다."""
        event.stop()
        self.post_message(self.OptionClicked(event.index))

    def hide(self) -> None:
        """팝업을 숨깁니다."""
        self._pending_suggestions = []
        self._rebuild_generation += 1  # Cancel any in-flight rebuild
        self.styles.display = "none"  # type: ignore[assignment]  # Textual accepts string display values at runtime

    def show(self) -> None:
        """팝업을 표시합니다."""
        self.styles.display = "block"


# ---------------------------------------------------------------------------
# Text entry behavior, history navigation, and paste heuristics
# ---------------------------------------------------------------------------

class ChatTextArea(TextArea):
    """채팅 입력을 위한 사용자 정의 키 처리가 포함된 TextArea 하위 클래스입니다."""

    BINDINGS: ClassVar[list[Binding]] = [
        Binding(
            "shift+enter,alt+enter,ctrl+enter",
            "insert_newline",
            "New Line",
            show=False,
            priority=True,
        ),
    ]
    """채팅 텍스트 영역에 대한 키 바인딩입니다.

    이는 바로 가기 키에 대한 단일 정보 소스입니다. `_NEWLINE_KEYS`은(는) 이 목록에서 파생되므로 `_on_key`은(는) 자동으로 동기화
    상태를 유지합니다.

    """

    _NEWLINE_KEYS: ClassVar[frozenset[str]] = frozenset(
        key
        for b in BINDINGS
        if b.action == "insert_newline"
        for key in b.key.split(",")
    )
    """`BINDINGS`에서 파생된 개행을 삽입하는 평면화된 키 세트입니다."""

    _skip_history_change_events: int
    """기록 기반 텍스트 교체 전에 카운터가 증가하므로
    결과 `TextArea.Changed` 이벤트(다음 메시지 루프 반복 시 발생)가 억제될 수 있습니다.
    `ChatInput.on_text_area_changed`은 카운터를 감소시킵니다.

    """

    _in_history: bool
    """사용자가 검색 기록을 보는 동안 `True`을 유지하는 영구 플래그입니다.

    위쪽/아래쪽 작업이 텍스트의 양쪽 끝에서 작동하도록 커서 경계 검사를 완화합니다.

    최신 항목을 지나 탐색하거나 제출하거나 삭제할 때 `False`로 재설정하세요.

    """

    class Submitted(Message):
        """텍스트가 제출되면 전송되는 메시지입니다."""

        def __init__(self, value: str) -> None:
            """제출된 값으로 초기화합니다."""
            self.value = value
            super().__init__()

    class HistoryPrevious(Message):
        """이전 이력 항목을 요청합니다."""

        def __init__(self, current_text: str) -> None:
            """저장을 위해 현재 텍스트로 초기화합니다."""
            self.current_text = current_text
            super().__init__()

    class HistoryNext(Message):
        """다음 기록 항목을 요청합니다."""

    class PastedPaths(Message):
        """붙여넣기 페이로드가 파일 경로로 확인될 때 전송되는 메시지입니다."""

        def __init__(self, raw_text: str, paths: list[Path]) -> None:
            """원시 붙여넣은 텍스트와 구문 분석된 파일 경로로 초기화합니다."""
            self.raw_text = raw_text
            self.paths = paths
            super().__init__()

    class Typing(Message):
        """사용자가 인쇄 가능한 키나 백스페이스를 누를 때 게시됩니다.

        앱이 타이핑 활동을 추적하기 위해 `ChatInput`에 의해 `ChatInput.Typing`로 전달됩니다.

        """

    def __init__(self, **kwargs: Any) -> None:
        """채팅 텍스트 영역을 초기화합니다."""
        # Remove placeholder if passed, TextArea doesn't support it the same way
        kwargs.pop("placeholder", None)
        super().__init__(**kwargs)
        self._skip_history_change_events = 0
        self._in_history = False
        self._completion_active = False
        # Buffer quote-prefixed high-frequency key bursts from terminals that
        # emulate paste via rapid key events instead of dispatching a paste
        # event.
        self._paste_burst_buffer = ""
        self._paste_burst_last_char_time: float | None = None
        self._paste_burst_timer: Timer | None = None
        # See _BACKSLASH_ENTER_GAP_SECONDS for context.
        self._backslash_pending_time: float | None = None

    def set_app_focus(self, *, has_focus: bool) -> None:
        """앱이 커서를 활성 상태로 표시할지 여부를 설정합니다.

        Args:
            has_focus: 앱 입력에 집중해야 하는지 여부입니다.

        """
        self._backslash_pending_time = None
        if has_focus and not self.has_focus:
            self.call_after_refresh(self.focus)

    def set_completion_active(self, *, active: bool) -> None:
        """완료 제안 표시 여부를 설정합니다."""
        self._completion_active = active

    def action_insert_newline(self) -> None:
        """개행 문자를 삽입합니다."""
        self.insert("\n")

    def _cancel_paste_burst_timer(self) -> None:
        """예약된 페이스트-버스트 플러시 타이머를 취소합니다."""
        if self._paste_burst_timer is None:
            return
        self._paste_burst_timer.stop()
        self._paste_burst_timer = None

    def _schedule_paste_burst_flush(self) -> None:
        """버퍼링된 붙여넣기 버스트 텍스트에 대한 유휴 시간 플러시를 예약합니다."""
        self._cancel_paste_burst_timer()
        self._paste_burst_timer = self.set_timer(
            _PASTE_BURST_FLUSH_DELAY_SECONDS, self._flush_paste_burst
        )

    def _start_paste_burst(self, char: str, now: float) -> None:
        """붙여넣기와 같은 키 입력 버스트 버퍼링을 시작합니다."""
        self._paste_burst_buffer = char
        self._paste_burst_last_char_time = now
        self._schedule_paste_burst_flush()

    def _append_paste_burst(self, text: str, now: float) -> None:
        """활성 붙여넣기-버스트 버퍼에 텍스트를 추가합니다."""
        if not self._paste_burst_buffer:
            self._start_paste_burst(text, now)
            return
        self._paste_burst_buffer += text
        self._paste_burst_last_char_time = now
        self._schedule_paste_burst_flush()

    def _should_start_paste_burst(self, char: str) -> bool:
        """키 누르기가 붙여넣기-버스트 버퍼링을 시작해야 하는지 여부를 반환합니다.

        빈 커서에서 따옴표가 붙은 입력으로 제한하면 일반 입력 및 슬래시 명령 입력에 대한 거짓 긍정이 줄어듭니다.

        """
        if char not in _PASTE_BURST_START_CHARS:
            return False
        if self.text or not self.selection.is_empty:
            return False
        row, col = self.cursor_location
        return row == 0 and col == 0

    async def _flush_paste_burst(self) -> None:
        """삭제된 경로 구문 분석을 통해 버퍼링된 버스트 텍스트를 플러시합니다.

        구문 분석에 실패하면 버퍼링된 텍스트가 변경되지 않고 삽입되므로 일반 입력 동작이 유지됩니다.

        """
        payload = self._paste_burst_buffer
        self._paste_burst_buffer = ""
        self._paste_burst_last_char_time = None
        self._cancel_paste_burst_timer()
        if not payload:
            return

        from deepagents_cli.input import parse_pasted_path_payload

        try:
            parsed = await asyncio.to_thread(parse_pasted_path_payload, payload)
        except Exception:  # noqa: BLE001  # Treat thread failure as non-path text
            parsed = None
        if parsed is not None:
            self.post_message(self.PastedPaths(payload, parsed.paths))
            return

        self.insert(payload)

    def _delete_preceding_backslash(self) -> bool:
        """커서 바로 앞의 백슬래시 문자를 삭제합니다.

        호출자는 이 위치에 백슬래시가 예상되는지 확인해야 합니다. 이 메서드는 문자를 삭제하기 전에 문자를 확인합니다.

        Returns:
            백슬래시가 발견되어 삭제된 경우 `True`, 그렇지 않은 경우 `False`.

        """
        row, col = self.cursor_location
        if col > 0:
            start = (row, col - 1)
            if self.document.get_text_range(start, self.cursor_location) == "\\":
                self.delete(start, self.cursor_location)
                return True
        elif row > 0:
            prev_line = self.document.get_line(row - 1)
            start = (row - 1, len(prev_line) - 1)
            end = (row - 1, len(prev_line))
            if self.document.get_text_range(start, end) == "\\":
                self.delete(start, self.cursor_location)
                return True
        return False

    async def _on_key(self, event: events.Key) -> None:
        """주요 이벤트를 처리합니다."""
        # VS Code 1.110 incorrectly sends space as a CSI u escape code
        # (`\x1b[32u`) instead of a plain ` ` character.  Textual parses
        # this as Key(key='space', character=None, is_printable=False), so
        # the TextArea never inserts the space.  Per the kitty keyboard
        # protocol spec, keys that generate text (like space) should NOT
        # use CSI u encoding — VS Code is the outlier here.
        #
        # This workaround should be safe to keep indefinitely: once VS Code or
        # Textual fixes the issue upstream, `character` will be `' '` and
        # this branch simply won't match.
        #
        # Upstream: https://github.com/Textualize/textual/issues/6408
        if event.key == "space" and event.character is None:
            event.prevent_default()
            event.stop()
            self.insert(" ")
            self.post_message(self.Typing())
            return

        now = time.monotonic()

        # Signal typing activity for printable keys and backspace so the app
        # can defer approval widgets while the user is actively editing.
        if event.is_printable or event.key == "backspace":
            self.post_message(self.Typing())

        if self._paste_burst_buffer:
            if event.key == "enter":
                self._append_paste_burst("\n", now)
                event.prevent_default()
                event.stop()
                return

            if event.is_printable and event.character is not None:
                last_time = self._paste_burst_last_char_time
                if (
                    last_time is not None
                    and (now - last_time) <= _PASTE_BURST_CHAR_GAP_SECONDS
                ):
                    self._append_paste_burst(event.character, now)
                    event.prevent_default()
                    event.stop()
                    return

            await self._flush_paste_burst()

        if (
            event.is_printable
            and event.character is not None
            and self._should_start_paste_burst(event.character)
        ):
            self._start_paste_burst(event.character, now)
            event.prevent_default()
            event.stop()
            return

        # Some terminals (e.g. VSCode built-in) send a literal backslash
        # followed by enter for shift+enter.  When enter arrives shortly
        # after a backslash, delete the backslash and insert a newline.
        if (
            event.key == "enter"
            and not self._completion_active
            and self._backslash_pending_time is not None
            and (now - self._backslash_pending_time) <= _BACKSLASH_ENTER_GAP_SECONDS
        ):
            self._backslash_pending_time = None
            if self._delete_preceding_backslash():
                event.prevent_default()
                event.stop()
                self.insert("\n")
                return
        self._backslash_pending_time = None

        if event.key == "backslash" and event.character == "\\":
            self._backslash_pending_time = now

        # Modifier+Enter inserts newline — keys derived from BINDINGS
        if event.key in self._NEWLINE_KEYS:
            event.prevent_default()
            event.stop()
            self.insert("\n")
            return

        if event.key == "backspace" and self._delete_image_placeholder(backwards=True):
            event.prevent_default()
            event.stop()
            return

        if event.key == "delete" and self._delete_image_placeholder(backwards=False):
            event.prevent_default()
            event.stop()
            return

        # If completion is active, let parent handle navigation keys
        if self._completion_active and event.key in {"up", "down", "tab", "enter"}:
            # Prevent TextArea's default behavior (e.g., Enter inserting newline)
            # but let event bubble to ChatInput for completion handling
            event.prevent_default()
            return

        # Plain Enter submits
        if event.key == "enter":
            event.prevent_default()
            event.stop()
            value = self.text.strip()
            if value:
                self.post_message(self.Submitted(value))
            return

        # Up/Down arrow: only navigate history at input boundaries.
        # Up requires cursor at position (0, 0); Down requires cursor at
        # the very end.  When already browsing history, either boundary
        # allows navigation in both directions.
        if event.key in {"up", "down"}:
            row, col = self.cursor_location
            text = self.text
            lines = text.split("\n")
            last_row = len(lines) - 1
            at_start = row == 0 and col == 0
            at_end = row == last_row and col == len(lines[last_row])
            navigate = (
                event.key == "up" and (at_start or (self._in_history and at_end))
            ) or (event.key == "down" and (at_end or (self._in_history and at_start)))

            if navigate:
                event.prevent_default()
                event.stop()
                if event.key == "up":
                    self.post_message(self.HistoryPrevious(self.text))
                else:
                    self.post_message(self.HistoryNext())
                return

        await super()._on_key(event)

    def _delete_image_placeholder(self, *, backwards: bool) -> bool:
        """한 번의 키 누르기로 전체 이미지 자리 표시자 토큰을 삭제합니다.

        Args:
            backwards: 삭제 작업이 뒤로(`backspace`)인지, 앞으로(`delete`)인지 여부입니다.

        Returns:
            `True` 자리표시자 토큰이 삭제된 경우.

        """
        if not self.text or not self.selection.is_empty:
            return False

        cursor_offset = self.document.get_index_from_location(self.cursor_location)  # type: ignore[attr-defined]  # Document has this method; DocumentBase stub is narrower
        span = self._find_image_placeholder_span(cursor_offset, backwards=backwards)
        if span is None:
            return False

        start, end = span
        start_location = self.document.get_location_from_index(start)  # type: ignore[attr-defined]  # Document has this method; DocumentBase stub is narrower
        end_location = self.document.get_location_from_index(end)  # type: ignore[attr-defined]
        self.delete(start_location, end_location)
        self.move_cursor(start_location)
        return True

    def _find_image_placeholder_span(
        self, cursor_offset: int, *, backwards: bool
    ) -> tuple[int, int] | None:
        """현재 커서 및 키 방향을 삭제할 자리 표시자 범위를 반환합니다.

        Args:
            cursor_offset: 텍스트 시작 부분부터 커서의 문자 오프셋입니다.
            backwards: 삭제 작업이 뒤로(백스페이스)인지 앞으로(삭제)인지 여부입니다.

        """
        text = self.text
        # Check both image and video placeholders
        for pattern in (IMAGE_PLACEHOLDER_PATTERN, VIDEO_PLACEHOLDER_PATTERN):
            for match in pattern.finditer(text):
                start, end = match.span()
                if backwards:
                    # Cursor is inside token or right after a trailing space inserted
                    # with the token.
                    if start < cursor_offset <= end:
                        return start, end
                    if cursor_offset > 0:
                        previous_index = cursor_offset - 1
                        if (
                            previous_index < len(text)
                            and previous_index == end
                            and text[previous_index].isspace()
                        ):
                            return start, cursor_offset
                elif start <= cursor_offset < end:
                    return start, end
        return None

    async def _on_paste(self, event: events.Paste) -> None:
        """붙여넣기 이벤트를 처리하고 드래그된 파일 경로를 감지합니다."""
        self._backslash_pending_time = None
        if self._paste_burst_buffer:
            await self._flush_paste_burst()

        from deepagents_cli.input import parse_pasted_path_payload

        try:
            parsed = await asyncio.to_thread(parse_pasted_path_payload, event.text)
        except Exception:  # noqa: BLE001  # Treat thread failure as non-path text
            parsed = None
        if parsed is None:
            # Don't call super() here — Textual's MRO dispatch already calls
            # TextArea._on_paste after this handler returns. Calling super()
            # would insert the text a second time, duplicating the paste.
            return

        event.prevent_default()
        event.stop()
        self.post_message(self.PastedPaths(event.text, parsed.paths))

    def set_text_from_history(self, text: str) -> None:
        """기록 탐색에서 텍스트를 설정합니다."""
        self._paste_burst_buffer = ""
        self._paste_burst_last_char_time = None
        self._cancel_paste_burst_timer()
        self._backslash_pending_time = None
        self._skip_history_change_events += 1
        self.text = text
        # Move cursor to end
        lines = text.split("\n")
        last_row = len(lines) - 1
        last_col = len(lines[last_row])
        self.move_cursor((last_row, last_col))

    def clear_text(self) -> None:
        """텍스트 영역을 지웁니다."""
        self._in_history = False
        # Increment (not reset) so any pending Changed event from a prior
        # set_text_from_history is still suppressed, plus one for the
        # self.text = "" assignment below.
        self._skip_history_change_events += 1
        self._paste_burst_buffer = ""
        self._paste_burst_last_char_time = None
        self._cancel_paste_burst_timer()
        self._backslash_pending_time = None
        self.text = ""
        self.move_cursor((0, 0))


# ---------------------------------------------------------------------------
# Adapter layer and top-level chat input container
# ---------------------------------------------------------------------------

class _CompletionViewAdapter:
    """완성 공간 대체를 텍스트 영역 좌표로 변환합니다."""

    def __init__(self, chat_input: ChatInput) -> None:
        """소유한 `ChatInput`을(를) 사용하여 어댑터를 초기화합니다."""
        self._chat_input = chat_input

    def render_completion_suggestions(
        self, suggestions: list[tuple[str, str]], selected_index: int
    ) -> None:
        """제안 렌더링을 `ChatInput`에 위임하세요."""
        self._chat_input.render_completion_suggestions(suggestions, selected_index)

    def clear_completion_suggestions(self) -> None:
        """완료 청산을 `ChatInput`에 위임하세요."""
        self._chat_input.clear_completion_suggestions()

    def replace_completion_range(self, start: int, end: int, replacement: str) -> None:
        """텍스트를 바꾸기 전에 완료 인덱스를 텍스트 영역 인덱스에 매핑합니다."""
        self._chat_input.replace_completion_range(
            self._chat_input._completion_index_to_text_index(start),
            self._chat_input._completion_index_to_text_index(end),
            replacement,
        )


class ChatInput(Vertical):
    """프롬프트, 여러 줄 텍스트, 자동 완성 및 기록이 포함된 채팅 입력 위젯입니다.

    기능: - TextArea를 사용한 여러 줄 입력 - 제출하려면 Enter, 개행용 수정자 키(`config.newline_shortcut` 참조) -
    입력 경계에서 명령 히스토리를 위한 위쪽/아래쪽 화살표(텍스트 시작/끝) - @(파일) 및 /(명령)에 대한 자동 완성

    """

    DEFAULT_CSS = """
    ChatInput {
        height: auto;
        min-height: 3;
        max-height: 25;
        padding: 0;
        background: $surface;
        border: solid $primary;
    }

    ChatInput.mode-shell {
        border: solid $mode-bash;
    }

    ChatInput.mode-command {
        border: solid $mode-command;
    }

    ChatInput .input-row {
        height: auto;
        width: 100%;
    }

    ChatInput .input-prompt {
        width: 3;
        height: 1;
        padding: 0 1;
        color: $primary;
        text-style: bold;
    }

    ChatInput.mode-shell .input-prompt {
        color: $mode-bash;
    }

    ChatInput.mode-command .input-prompt {
        color: $mode-command;
    }

    ChatInput ChatTextArea {
        width: 1fr;
        height: auto;
        min-height: 1;
        max-height: 8;
        border: none;
        background: transparent;
        padding: 0;
    }

    ChatInput ChatTextArea:focus {
        border: none;
    }
    """
    """즉각적인 시각적 피드백을 위해 테두리 및 프롬프트 글리프 색상이 모드별로 변경됩니다."""

    class Submitted(Message):
        """입력이 제출되면 전송되는 메시지입니다."""

        def __init__(self, value: str, mode: str = "normal") -> None:
            """값과 모드로 초기화합니다."""
            super().__init__()
            self.value = value
            self.mode = mode

    class ModeChanged(Message):
        """입력 모드가 변경되면 전송되는 메시지입니다."""

        def __init__(self, mode: str) -> None:
            """새로운 모드로 초기화하세요."""
            super().__init__()
            self.mode = mode

    class Typing(Message):
        """사용자가 입력에서 인쇄 가능한 키나 백스페이스를 누를 때 게시됩니다.

        앱은 이를 사용하여 사용자가 적극적으로 입력하는 동안 승인 위젯을 지연시켜 실수로 키를 누르는 것(예: `y`, `n`)이 승인 결정을
        트리거하는 것을 방지합니다.

        """

    mode: reactive[str] = reactive("normal")

    def __init__(
        self,
        cwd: str | Path | None = None,
        history_file: Path | None = None,
        image_tracker: MediaTracker | None = None,
        **kwargs: Any,
    ) -> None:
        """채팅 입력 위젯을 초기화합니다.

        Args:
            cwd: 파일 완성을 위한 현재 작업 디렉터리
            history_file: 기록 파일 경로(기본값: ~/.deepagents/history.jsonl)
            image_tracker: 첨부된 이미지에 대한 선택적 추적기
            **kwargs: 부모에 대한 추가 인수

        """
        super().__init__(**kwargs)
        self._cwd = Path(cwd) if cwd else Path.cwd()
        self._image_tracker = image_tracker
        self._text_area: ChatTextArea | None = None
        self._popup: CompletionPopup | None = None
        self._completion_manager: MultiCompletionManager | None = None
        self._completion_view: _CompletionViewAdapter | None = None
        self._slash_controller: SlashCommandController | None = None

        # Guard flag: set True before programmatically stripping the mode
        # prefix character so the resulting text-change event does not
        # re-evaluate mode.
        self._stripping_prefix = False

        # When the user submits, we clear the text area which fires a
        # text-change event. Without this guard the tracker would see the
        # now-empty text, assume all media were deleted, and discard them
        # before the app has a chance to send them. Each submit bumps the
        # counter by one; the next text-change event decrements it and
        # skips the sync.
        self._skip_media_sync_events = 0

        # Number of virtual prefix characters currently injected for
        # completion controller calls (0 for normal, 1 for shell/command).
        self._completion_prefix_len = 0

        # Guard flag: set while replacing a dropped path payload with an
        # inline image placeholder so the resulting change event doesn't
        # immediately recurse into the same replacement path.
        self._applying_inline_path_replacement = False

        # Track current suggestions for click handling
        self._current_suggestions: list[tuple[str, str]] = []
        self._current_selected_index = 0

        # Set up history manager
        if history_file is None:
            history_file = _default_history_path()
        self._history = HistoryManager(history_file)

    def compose(self) -> ComposeResult:  # noqa: PLR6301  # Textual widget method convention
        """채팅 입력 레이아웃을 구성합니다.

        Yields:
            입력 행 및 완료 팝업을 위한 위젯입니다.

        """
        with Horizontal(classes="input-row"):
            yield Static(">", classes="input-prompt", id="prompt")
            yield ChatTextArea(id="chat-input")

        yield CompletionPopup(id="completion-popup")

    def on_mount(self) -> None:
        """마운트 후 구성요소를 초기화합니다."""
        if is_ascii_mode():
            colors = theme.get_theme_colors(self)
            self.styles.border = ("ascii", colors.primary)

        self._text_area = self.query_one("#chat-input", ChatTextArea)
        self._popup = self.query_one("#completion-popup", CompletionPopup)

        # Both controllers implement the CompletionController protocol but have
        # different concrete types; the list-item warning is a false positive.
        self._completion_view = _CompletionViewAdapter(self)
        self._file_controller = FuzzyFileController(
            self._completion_view, cwd=self._cwd
        )
        self._slash_controller = SlashCommandController(
            SLASH_COMMANDS, self._completion_view
        )
        self._completion_manager = MultiCompletionManager(
            [
                self._slash_controller,
                self._file_controller,
            ]  # type: ignore[list-item]  # Controller types are compatible at runtime
        )

        self.run_worker(
            self._file_controller.warm_cache(),
            exclusive=False,
            exit_on_error=False,
        )
        self._text_area.focus()

    def update_slash_commands(self, commands: list[tuple[str, str, str]]) -> None:
        """슬래시 명령 컨트롤러의 명령 목록을 업데이트합니다.

        정적 명령을 동적 `/skill:` 항목과 병합하는 기술을 발견한 후 앱에서 호출됩니다.

        Args:
            commands: `(command, description, hidden_keywords)` 튜플의 전체 목록입니다.

        """
        if self._slash_controller:
            self._slash_controller.update_commands(commands)
        else:
            logger.warning(
                "Cannot update slash commands: controller not initialized "
                "(widget not yet mounted)"
            )

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """입력 모드 및 업데이트 완료를 감지합니다."""
        text = event.text_area.text
        self._sync_media_tracker_to_text(text)

        # History handlers explicitly decide mode and stripped display text.
        # Skip mode detection here so recalled entries don't inherit stale mode.
        if self._text_area and self._text_area._skip_history_change_events > 0:
            self._text_area._skip_history_change_events -= 1
            if self._completion_manager:
                self._completion_manager.reset()
            self.scroll_visible()
            return
        if self._text_area and self._text_area._skip_history_change_events < 0:
            logger.warning(
                "_skip_history_change_events is negative (%d); resetting to 0",
                self._text_area._skip_history_change_events,
            )
            self._text_area._skip_history_change_events = 0

        if self._applying_inline_path_replacement:
            self._applying_inline_path_replacement = False
        elif self._apply_inline_dropped_path_replacement(text):
            return

        # Checked after the guards above so we skip the (potentially slow)
        # filesystem lookup when the text change came from history navigation
        # or prefix stripping, which never need path detection.
        is_path_payload = self._is_dropped_path_payload(text)

        # Guard: skip mode re-detection after we programmatically stripped
        # a prefix character.
        if self._stripping_prefix:
            self._stripping_prefix = False
        elif text and text[0] in PREFIX_TO_MODE:
            if text[0] == "/" and is_path_payload:
                # Absolute dropped paths stay normal input, not slash-command mode.
                if self.mode != "normal":
                    self.mode = "normal"
            else:
                # Detected a mode-trigger prefix (e.g. "!" or "/").
                # Strip it unconditionally -- even when already in the correct
                # mode -- because completion controllers may write replacement
                # text that re-includes the trigger character.  The
                # _stripping_prefix guard prevents the resulting change event
                # from looping back here.
                detected = PREFIX_TO_MODE[text[0]]
                if self.mode != detected:
                    self.mode = detected
                self._strip_mode_prefix()
                # Fall through to update completion suggestions in the same
                # refresh cycle as the mode/glyph change rather than waiting
                # for the next text-change event caused by the prefix strip.
                # Note: the strip's text-change event will also call
                # on_text_changed (idempotently) since _stripping_prefix only
                # skips mode detection, not the completion block below.
        # Update completion suggestions using completion-space text/cursor.
        if self._completion_manager and self._text_area:
            if is_path_payload:
                self._completion_manager.reset()
            else:
                vtext, vcursor = self._completion_text_and_cursor()
                self._completion_manager.on_text_changed(vtext, vcursor)

        # Scroll input into view when content changes (handles text wrap)
        self.scroll_visible()

    @staticmethod
    def _parse_dropped_path_payload(
        text: str, *, allow_leading_path: bool = False
    ) -> ParsedPastedPathPayload | None:
        """단일 파서 진입점을 통해 삭제된 경로 페이로드 텍스트를 구문 분석합니다.

        Returns:
            파싱된 페이로드 세부정보, 그렇지 않으면 `None`.

        """
        from deepagents_cli.input import parse_pasted_path_payload

        return parse_pasted_path_payload(text, allow_leading_path=allow_leading_path)

    def _parse_dropped_path_payload_with_command_recovery(
        self, text: str, *, allow_leading_path: bool = False
    ) -> tuple[str, ParsedPastedPathPayload | None]:
        """명령 모드에서 페이로드를 구문 분석하고 제거된 선행 슬래시를 복구합니다.

        Args:
            text: 구문 분석할 텍스트를 입력합니다.
            allow_leading_path: 선행 경로 + 접미사 페이로드를 구문 분석할지 여부입니다.

        Returns:
            `(candidate_text, parsed_payload)`의 튜플입니다.

        """
        candidate = text
        parsed = self._parse_dropped_path_payload(
            text, allow_leading_path=allow_leading_path
        )
        if parsed is not None:
            return candidate, parsed

        if self.mode != "command":
            return candidate, None

        prefixed = f"/{text.lstrip('/')}"
        parsed = self._parse_dropped_path_payload(
            prefixed, allow_leading_path=allow_leading_path
        )
        if parsed is None:
            return candidate, None

        logger.debug(
            "Recovering stripped absolute path; resetting mode from "
            "'command' to 'normal'"
        )
        self.mode = "normal"
        return prefixed, parsed

    def _extract_leading_dropped_path_with_command_recovery(
        self, text: str
    ) -> tuple[str, tuple[Path, int] | None]:
        """명령 모드 복구를 사용하여 선행 삭제 경로 토큰을 추출합니다.

        Args:
            text: 구문 분석할 텍스트를 입력합니다.

        Returns:
            `(candidate_text, leading_match)`의 튜플. 여기서 `leading_match`은 추출이 성공하면 `(path,
            token_end)`이고, 그렇지 않으면 `None`입니다.

        """
        from deepagents_cli.input import extract_leading_pasted_file_path

        leading_match = extract_leading_pasted_file_path(text)
        candidate = text
        if leading_match is not None:
            return candidate, leading_match

        if self.mode != "command":
            return candidate, None

        prefixed = f"/{text.lstrip('/')}"
        leading_match = extract_leading_pasted_file_path(prefixed)
        if leading_match is None:
            return candidate, None

        logger.debug(
            "Recovering stripped absolute leading path; resetting mode "
            "from 'command' to 'normal'"
        )
        self.mode = "normal"
        return prefixed, leading_match

    @staticmethod
    def _is_existing_path_payload(text: str) -> bool:
        """텍스트가 기존 파일에 대한 삭제된 경로 페이로드인지 여부를 반환합니다."""
        if len(text) < 2:  # noqa: PLR2004  # Need at least '/' + one char
            return False
        from deepagents_cli.input import parse_pasted_path_payload

        return parse_pasted_path_payload(text, allow_leading_path=True) is not None

    def _is_dropped_path_payload(self, text: str) -> bool:
        """현재 텍스트가 삭제된 파일 경로 페이로드처럼 보이는지 여부를 반환합니다."""
        if not text:
            return False
        if self._is_existing_path_payload(text):
            return True
        if self.mode == "command":
            candidate = f"/{text.lstrip('/')}"
            return self._is_existing_path_payload(candidate)
        return False

    def _strip_mode_prefix(self) -> None:
        """텍스트 영역에서 첫 번째 문자(모드 트리거)를 제거합니다.

        결과 텍스트 변경 이벤트가 새 입력으로 잘못 해석되지 않도록 `_stripping_prefix` 가드를 설정합니다.

        """
        if not self._text_area:
            return
        if self._stripping_prefix:
            logger.warning(
                "Previous _stripping_prefix guard was never cleared; "
                "resetting. This may indicate a missed text-change event."
            )
        text = self._text_area.text
        if not text:
            return
        row, col = self._text_area.cursor_location
        self._stripping_prefix = True
        self._text_area.text = text[1:]
        if row == 0 and col > 0:
            col -= 1
        self._text_area.move_cursor((row, col))

    def _completion_text_and_cursor(self) -> tuple[str, int]:
        """완료 공간에 컨트롤러 쪽 텍스트/커서를 반환합니다.

        또한 `_completion_index_to_text_index`에 대한 후속 호출이 일치하는 오프셋을 사용하도록
        `_completion_prefix_len`을 업데이트합니다.

        """
        if not self._text_area:
            self._completion_prefix_len = 0
            return "", 0

        text = self._text_area.text
        cursor = self._get_cursor_offset()
        prefix = MODE_PREFIXES.get(self.mode, "")
        self._completion_prefix_len = len(prefix)

        if prefix:
            return prefix + text, cursor + len(prefix)
        return text, cursor

    def _completion_index_to_text_index(self, index: int) -> int:
        """완성 공간 인덱스를 텍스트 영역 인덱스로 변환합니다.

        Args:
            index: 완성 공간의 커서/인덱스 위치.

        Returns:
            텍스트 영역 공간에 고정된 인덱스입니다.

        """
        if not self._text_area:
            return 0

        mapped = index - self._completion_prefix_len
        text_len = len(self._text_area.text)
        if mapped < 0 or mapped > text_len:
            logger.warning(
                "Completion index %d mapped to %d, outside [0, %d]; "
                "clamping (prefix_len=%d, mode=%s)",
                index,
                mapped,
                text_len,
                self._completion_prefix_len,
                self.mode,
            )
        return max(0, min(mapped, text_len))

    def _submit_value(self, value: str) -> None:
        """모드 접두어 추가, 기록에 저장, 메시지 게시 및 입력 재설정.

        이는 모든 제출 흐름에 대한 단일 경로이므로 접두사 앞에 추가 + 기록 + 게시 + 지우기 + 모드 재설정 논리가 한 곳에 유지됩니다.

        Args:
            value: 제출할 제거된 텍스트입니다(모드 접두사 제외).

        """
        if not value:
            return

        if self._completion_manager:
            self._completion_manager.reset()

        value = self._replace_submitted_paths_with_images(value)

        # Prepend mode prefix so the app layer receives the original trigger
        # form (e.g. "!ls", "/help"). The value may already contain the prefix
        # when a completion controller wrote it back into the text area before
        # the strip handler ran.
        prefix = MODE_PREFIXES.get(self.mode, "")
        if prefix and not value.startswith(prefix):
            value = prefix + value

        self._history.add(value)
        self.post_message(self.Submitted(value, self.mode))

        if self._text_area:
            # Preserve submission-time attachments until adapter consumes them.
            self._skip_media_sync_events += 1
            self._text_area.clear_text()
        self.mode = "normal"

    def _sync_media_tracker_to_text(self, text: str) -> None:
        """추적된 미디어를 입력 텍스트의 자리 표시자 토큰과 정렬되도록 유지합니다.

        Args:
            text: 입력 영역의 현재 텍스트입니다.

        """
        if not self._image_tracker:
            return
        if self._skip_media_sync_events:
            if self._skip_media_sync_events < 0:
                logger.warning(
                    "_skip_media_sync_events is negative (%d); resetting to 0",
                    self._skip_media_sync_events,
                )
                self._skip_media_sync_events = 0
            else:
                self._skip_media_sync_events -= 1
            return
        self._image_tracker.sync_to_text(text)

    def on_chat_text_area_typing(
        self,
        event: ChatTextArea.Typing,  # noqa: ARG002  # Textual event handler signature
    ) -> None:
        """타이핑 활동을 앱에 `ChatInput.Typing`로 전달합니다."""
        self.post_message(self.Typing())

    def on_chat_text_area_submitted(self, event: ChatTextArea.Submitted) -> None:
        """텍스트 제출을 처리합니다.

        항상 Submitted 이벤트를 게시합니다. 앱 계층은 에이전트 상태에 따라 즉시 처리할지 아니면 대기열에 추가할지 결정합니다.

        """
        self._submit_value(event.value)

    def on_chat_text_area_history_previous(
        self, event: ChatTextArea.HistoryPrevious
    ) -> None:
        """이전 요청 내역을 처리합니다."""
        entry = self._history.get_previous(event.current_text, query=event.current_text)
        if entry is not None and self._text_area:
            mode, display_text = self._history_entry_mode_and_text(entry)
            self.mode = mode
            self._text_area.set_text_from_history(display_text)
        # No-match path: don't reset the counter — a pending Changed event
        # from a prior set_text_from_history call may still be in flight.
        # Keep text area's _in_history in sync with the history manager.
        if self._text_area:
            self._text_area._in_history = self._history.in_history

    def on_chat_text_area_history_next(
        self,
        event: ChatTextArea.HistoryNext,  # noqa: ARG002  # Textual event handler signature
    ) -> None:
        """다음 요청 기록을 처리합니다."""
        entry = self._history.get_next()
        if entry is not None and self._text_area:
            mode, display_text = self._history_entry_mode_and_text(entry)
            self.mode = mode
            self._text_area.set_text_from_history(display_text)
        # No-match path: don't reset the counter — a pending Changed event
        # from a prior set_text_from_history call may still be in flight.
        # Keep text area's _in_history in sync with the history manager.
        # When the user presses Down past the newest entry, get_next()
        # resets navigation internally, so in_history becomes False.
        if self._text_area:
            self._text_area._in_history = self._history.in_history

    def on_chat_text_area_pasted_paths(self, event: ChatTextArea.PastedPaths) -> None:
        """삭제된 파일 경로를 해결하는 붙여넣기 페이로드를 처리합니다."""
        if not self._text_area:
            return

        self._insert_pasted_paths(event.raw_text, event.paths)

    def handle_external_paste(self, pasted: str) -> bool:
        """입력에 초점이 맞춰지지 않은 경우 앱 수준 라우팅에서 붙여넣은 텍스트를 처리합니다.

        텍스트 영역이 마운트되면 붙여넣기가 항상 소비됩니다. 파일 경로는 이미지로 첨부되고 일반 텍스트가 직접 삽입됩니다.

        Args:
            pasted: 원시 붙여넣은 텍스트 페이로드입니다.

        Returns:
            `True` 텍스트 영역이 마운트되고 붙여넣기가 삽입되었을 때,
                `False` 위젯이 아직 구성되지 않은 경우.

        """
        if not self._text_area:
            return False

        parsed = self._parse_dropped_path_payload(pasted)
        if parsed is None:
            self._text_area.insert(pasted)
        else:
            self._insert_pasted_paths(pasted, parsed.paths)

        self._text_area.focus()
        return True

    def _apply_inline_dropped_path_replacement(self, text: str) -> bool:
        """전체 삭제 경로 페이로드 텍스트를 이미지 자리 표시자로 바꿉니다.

        일부 터미널에서는 전용 붙여넣기 이벤트를 전달하는 대신 드래그 앤 드롭 페이로드를 일반 텍스트로 삽입합니다. 현재 텍스트가 하나 이상의 파일
        경로로 확인되고 하나 이상의 경로가 이미지인 경우 텍스트를 `[image N]` 자리 표시자에 인라인으로 다시 작성합니다.

        Args:
            text: 현재 텍스트 영역 콘텐츠입니다.

        Returns:
            텍스트가 인라인으로 다시 작성된 경우 `True`, 그렇지 않은 경우 `False`.

        """
        if not self._text_area:
            return False

        parsed = self._parse_dropped_path_payload(text)
        if parsed is None:
            return False

        replacement, attached = self._build_path_replacement(
            text, parsed.paths, add_trailing_space=True
        )
        if not attached or replacement == text:
            return False

        self._applying_inline_path_replacement = True
        self._text_area.text = replacement
        lines = replacement.split("\n")
        self._text_area.move_cursor((len(lines) - 1, len(lines[-1])))
        return True

    def _insert_pasted_paths(self, raw_text: str, paths: list[Path]) -> None:
        """붙여넣은 경로 페이로드를 삽입하고 가능하면 이미지를 첨부하세요.

        Args:
            raw_text: 원본 붙여넣기 페이로드 텍스트.
            paths: 페이로드에서 구문 분석된 파일 경로가 확인되었습니다.

        """
        if not self._text_area:
            return
        replacement, attached = self._build_path_replacement(
            raw_text, paths, add_trailing_space=True
        )
        if attached:
            self._text_area.insert(replacement)
            return
        self._text_area.insert(raw_text)

    def _build_path_replacement(
        self,
        raw_text: str,
        paths: list[Path],
        *,
        add_trailing_space: bool,
    ) -> tuple[str, bool]:
        """삭제된 경로에 대한 대체 텍스트를 작성하고 이미지를 첨부하세요.

        Args:
            raw_text: 원본 붙여넣기 페이로드 텍스트.
            paths: 페이로드에서 구문 분석된 파일 경로가 확인되었습니다.
            add_trailing_space: 경로가 공백으로 구분된 경우 마지막 토큰 뒤에 후행 공백을 추가할지 여부입니다.

        Returns:
            `(replacement, attached)`의 튜플 여기서 `attached`은 하나 이상의 미디어 첨부 파일(이미지 또는 비디오)이
            생성되었는지 여부를 나타냅니다.

        """
        if not self._image_tracker:
            return raw_text, False

        from deepagents_cli.media_utils import (
            IMAGE_EXTENSIONS,
            MAX_MEDIA_BYTES,
            VIDEO_EXTENSIONS,
            ImageData,
            get_media_from_path,
        )

        parts: list[str] = []
        attached = False
        for path in paths:
            media = get_media_from_path(path)
            if media is not None:
                kind = "image" if isinstance(media, ImageData) else "video"
                parts.append(self._image_tracker.add_media(media, kind))
                attached = True
                continue

            # Check if it looked like media but failed validation
            suffix = path.suffix.lower()
            if suffix in IMAGE_EXTENSIONS or suffix in VIDEO_EXTENSIONS:
                label = "Video" if suffix in VIDEO_EXTENSIONS else "Image"
                try:
                    size = path.stat().st_size
                    if size > MAX_MEDIA_BYTES:
                        msg = (
                            f"{label} too large: {path.name} "
                            f"({size // (1024 * 1024)} MB, max "
                            f"{MAX_MEDIA_BYTES // (1024 * 1024)} MB)"
                        )
                    else:
                        msg = f"Could not attach {label.lower()}: {path.name}"
                except OSError as exc:
                    logger.debug("Failed to stat media file %s: %s", path, exc)
                    msg = f"Could not attach {label.lower()}: {path.name}"
                self.app.notify(msg, severity="warning", timeout=5, markup=False)

            # Not a supported media file, keep as path
            logger.debug("Could not load media from dropped path: %s", path)
            parts.append(str(path))

        if not attached:
            return raw_text, False

        separator = "\n" if "\n" in raw_text else " "
        replacement = separator.join(parts)
        if separator == " " and add_trailing_space:
            replacement += " "
        return replacement, True

    def _replace_submitted_paths_with_images(self, value: str) -> str:
        """제출된 텍스트의 삭제된 경로 페이로드를 이미지 자리 표시자로 바꿉니다.

        전체 경로 페이로드와 접미사가 있는 선행 경로 페이로드(예: `'<path>' what is this?`)를 모두 처리합니다. 명령 모드가
        이전에 선행 슬래시를 제거한 경우 이 방법은 포기하기 전에 복원된 슬래시로 다시 시도합니다.

        Args:
            value: 제출된 텍스트를 제거했습니다(모드 접두사 없음).

        Returns:
            첨부가 성공하면 이미지 자리 표시자와 함께 텍스트가 제출되었습니다.

        """
        candidate, parsed = self._parse_dropped_path_payload_with_command_recovery(
            value, allow_leading_path=True
        )
        if parsed is None:
            return value

        if parsed.token_end is None:
            replacement, attached = self._build_path_replacement(
                candidate, parsed.paths, add_trailing_space=False
            )
            if attached:
                return replacement.strip()
            # Even when full-payload parsing resolves, still retry explicit
            # leading-token extraction before giving up.
            candidate, leading_match = (
                self._extract_leading_dropped_path_with_command_recovery(value)
            )
            if leading_match is None:
                return value
            leading_path, token_end = leading_match
        else:
            leading_path = parsed.paths[0]
            token_end = parsed.token_end

        replacement, attached = self._build_path_replacement(
            str(leading_path), [leading_path], add_trailing_space=False
        )
        if attached:
            suffix = candidate[token_end:].lstrip()
            if suffix:
                return f"{replacement.strip()} {suffix}".strip()
            return replacement.strip()
        return value

    @staticmethod
    def _history_entry_mode_and_text(entry: str) -> tuple[str, str]:
        """기록 항목에 대한 반환 모드 및 제거된 표시 텍스트입니다.

        Args:
            entry: 기록 저장소에서 읽은 원시 항목 값입니다.

        Returns:
            모드 트리거 접두사가 다음과 같은 `(mode, display_text)`의 튜플
                `display_text`에서 삭제되었습니다.

        """
        for prefix, mode in PREFIX_TO_MODE.items():
            # Small dict; loop is fine. No need to over-engineer right now
            if entry.startswith(prefix):
                return mode, entry[len(prefix) :]
        return "normal", entry

    async def on_key(self, event: events.Key) -> None:
        """완료 탐색을 위한 주요 이벤트를 처리합니다."""
        if not self._completion_manager or not self._text_area:
            return

        # Backspace at cursor position 0 (or on empty input) exits the
        # current mode (e.g. command/shell).  When the cursor is at the very
        # start of the text area, backspace is a no-op for the underlying
        # widget, so without this guard the user would be stuck in the mode.
        if (
            event.key == "backspace"
            and self.mode != "normal"
            and self._get_cursor_offset() == 0
        ):
            # Defer the popup reset so it coalesces with the glyph update
            # that watch_mode schedules via call_after_refresh.
            def _deferred_reset() -> None:
                if self._completion_manager is not None:
                    self._completion_manager.reset()

            self.call_after_refresh(_deferred_reset)
            self.mode = "normal"
            event.prevent_default()
            event.stop()
            return

        text, cursor = self._completion_text_and_cursor()
        result = self._completion_manager.on_key(event, text, cursor)

        match result:
            case CompletionResult.HANDLED:
                event.prevent_default()
                event.stop()
            case CompletionResult.SUBMIT:
                event.prevent_default()
                event.stop()
                self._submit_value(self._text_area.text.strip())
            case CompletionResult.IGNORED if event.key == "enter":
                # Handle Enter when completion is not active (shell/normal modes)
                value = self._text_area.text.strip()
                if value:
                    event.prevent_default()
                    event.stop()
                    self._submit_value(value)

    def _get_cursor_offset(self) -> int:
        """커서 오프셋을 단일 정수로 가져옵니다.

        Returns:
            텍스트 시작 부분부터의 문자 오프셋에 따른 커서 위치입니다.

        """
        if not self._text_area:
            return 0

        text = self._text_area.text
        row, col = self._text_area.cursor_location

        if not text:
            return 0

        lines = text.split("\n")
        row = max(0, min(row, len(lines) - 1))
        col = max(0, col)

        offset = sum(len(lines[i]) + 1 for i in range(row))
        return offset + min(col, len(lines[row]))

    def watch_mode(self, mode: str) -> None:
        """게시 모드가 변경된 메시지와 업데이트 프롬프트 표시기입니다.

        프롬프트 글리프 업데이트는 `call_after_refresh`을 통해 지연되므로 지연된 작업(예: 완료 팝업)을 예약하는 호출자가 두 시각적
        변경 사항을 단일 새로 고침으로 통합할 수 있습니다.

        """
        glyph = MODE_DISPLAY_GLYPHS.get(mode)
        if not glyph and mode != "normal":
            logger.warning(
                "No display glyph for mode %r; falling back to '>'",
                mode,
            )

        def _apply() -> None:
            self.remove_class("mode-shell", "mode-command")
            if glyph:
                self.add_class(f"mode-{mode}")
            try:
                prompt = self.query_one("#prompt", Static)
            except NoMatches:
                logger.warning("watch_mode._apply: #prompt widget not found")
                return
            prompt.update(glyph or ">")

        self.call_after_refresh(_apply)
        self.post_message(self.ModeChanged(mode))

    def focus_input(self) -> None:
        """입력 필드에 초점을 맞춥니다."""
        if self._text_area:
            self._text_area.focus()

    @property
    def value(self) -> str:
        """현재 입력 값을 가져옵니다.

        Returns:
            입력 필드의 현재 텍스트입니다.

        """
        if self._text_area:
            return self._text_area.text
        return ""

    @value.setter
    def value(self, val: str) -> None:
        """입력값을 설정합니다."""
        if self._text_area:
            self._text_area.text = val

    @property
    def input_widget(self) -> ChatTextArea | None:
        """기본 TextArea 위젯을 가져옵니다.

        Returns:
            ChatTextArea 위젯 또는 마운트되지 않은 경우 None입니다.

        """
        return self._text_area

    def set_disabled(self, *, disabled: bool) -> None:
        """입력 위젯을 활성화하거나 비활성화합니다."""
        if self._text_area:
            self._text_area.disabled = disabled
            if disabled:
                self._text_area.blur()
                if self._completion_manager:
                    self._completion_manager.reset()

    def set_cursor_active(self, *, active: bool) -> None:
        """입력 포커스 상태를 전환합니다(예: 에이전트가 작업하는 동안 포커스 해제).

        Args:
            active: 입력에 집중하고 입력을 수용해야 하는지 여부입니다.

        """
        if self._text_area:
            self._text_area.set_app_focus(has_focus=active)

    def exit_mode(self) -> bool:
        """현재 입력 모드(명령어/셸)를 종료하고 다시 정상으로 돌아갑니다.

        Returns:
            모드가 비정상이고 재설정된 경우 참입니다.

        """
        if self.mode == "normal":
            return False
        self.mode = "normal"
        if self._completion_manager:
            self._completion_manager.reset()
        self.clear_completion_suggestions()
        return True

    def dismiss_completion(self) -> bool:
        """완료 취소: 보기를 지우고 컨트롤러 상태를 재설정합니다.

        Returns:
            완료가 활성 상태이고 해제된 경우 True입니다.

        """
        if not self._current_suggestions:
            return False
        if self._completion_manager:
            self._completion_manager.reset()
        # Always clear local state so the popup is hidden even if the
        # manager's active controller was already None (no-op reset).
        self.clear_completion_suggestions()
        return True

    # =========================================================================
    # CompletionView protocol implementation
    # =========================================================================

    def render_completion_suggestions(
        self, suggestions: list[tuple[str, str]], selected_index: int
    ) -> None:
        """팝업에 렌더링 완료 제안이 표시됩니다."""
        prev_suggestions = self._current_suggestions
        self._current_suggestions = suggestions
        self._current_selected_index = selected_index

        if self._popup:
            # If only the selection changed (same items), skip full rebuild
            if suggestions == prev_suggestions:
                self._popup.update_selection(selected_index)
            else:
                self._popup.update_suggestions(suggestions, selected_index)
        # Tell TextArea that completion is active so it yields navigation keys
        if self._text_area:
            self._text_area.set_completion_active(active=bool(suggestions))

    def clear_completion_suggestions(self) -> None:
        """완료 팝업을 지우거나 숨깁니다."""
        self._current_suggestions = []
        self._current_selected_index = 0

        if self._popup:
            self._popup.hide()
        # Tell TextArea that completion is no longer active
        if self._text_area:
            self._text_area.set_completion_active(active=False)

    def on_completion_popup_option_clicked(
        self, event: CompletionPopup.OptionClicked
    ) -> None:
        """완료 옵션을 클릭하여 처리합니다."""
        if not self._current_suggestions or not self._text_area:
            return

        index = event.index
        if index < 0 or index >= len(self._current_suggestions):
            return

        # Get the selected completion
        label, _ = self._current_suggestions[index]
        text = self._text_area.text
        cursor = self._get_cursor_offset()

        # Determine replacement range based on completion type.
        # Slash completions use completion-space coordinates and are translated
        # through the completion view adapter.
        if label.startswith("/"):
            if self._completion_view is None:
                logger.warning(
                    "Slash completion clicked but _completion_view is not "
                    "initialized; this indicates a widget lifecycle issue."
                )
                return
            _, virtual_cursor = self._completion_text_and_cursor()
            self._completion_view.replace_completion_range(0, virtual_cursor, label)
        elif label.startswith("@"):
            # File mention: replace from @ to cursor
            at_index = text[:cursor].rfind("@")
            if at_index >= 0:
                self.replace_completion_range(at_index, cursor, label)

        # Reset completion state
        if self._completion_manager:
            self._completion_manager.reset()

        # Re-focus the text input after click
        self._text_area.focus()

    def replace_completion_range(self, start: int, end: int, replacement: str) -> None:
        """입력 필드의 텍스트를 바꿉니다."""
        if not self._text_area:
            return

        text = self._text_area.text

        start = max(0, min(start, len(text)))
        end = max(start, min(end, len(text)))

        prefix = text[:start]
        suffix = text[end:]

        # Add space after completion unless it's a directory path
        if replacement.endswith("/"):
            insertion = replacement
        else:
            insertion = replacement + " " if not suffix.startswith(" ") else replacement

        new_text = f"{prefix}{insertion}{suffix}"
        self._text_area.text = new_text

        # Calculate new cursor position and move cursor
        new_offset = start + len(insertion)
        lines = new_text.split("\n")
        remaining = new_offset
        for row, line in enumerate(lines):
            if remaining <= len(line):
                self._text_area.move_cursor((row, remaining))
                break
            remaining -= len(line) + 1
