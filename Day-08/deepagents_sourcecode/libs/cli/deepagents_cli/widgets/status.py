"""현재 CLI 세션 상태를 요약하는 바닥글 위젯입니다.

상태 표시줄에는 채팅 중에 계속 표시되어야 하는 모델 선택, 스레드 정보, 토큰 컨텍스트 및 기타 압축 세션 표시기가 표시됩니다.
"""

from __future__ import annotations

import logging
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

from textual.containers import Horizontal
from textual.content import Content
from textual.css.query import NoMatches
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static

from deepagents_cli.config import get_glyphs

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from textual import events
    from textual.app import ComposeResult, RenderResult
    from textual.geometry import Size


class ModelLabel(Widget):
    """스마트 잘림을 사용하여 오른쪽 정렬된 모델 이름을 표시하는 레이블입니다.

    전체 `provider:model` 텍스트가 맞지 않으면 공급자가 먼저 삭제됩니다. 기본 모델 이름이 여전히 맞지 않으면 가장 독특한 꼬리가 계속
    표시되도록 선행 줄임표를 사용하여 왼쪽이 잘립니다.

    """

    provider: reactive[str] = reactive("", layout=True)
    model: reactive[str] = reactive("", layout=True)

    def get_content_width(self, container: Size, viewport: Size) -> int:  # noqa: ARG002
        """`width: auto`이 작동하도록 고유 너비를 반환합니다.

Args:
            container: 컨테이너의 크기.
            viewport: 뷰포트의 크기.

Returns:
            Character length of the full provider: 모델 문자열.

        """
        if not self.model:
            return 0
        full = f"{self.provider}:{self.model}" if self.provider else self.model
        return len(full)

    def render(self) -> RenderResult:
        """너비 인식 잘림을 사용하여 모델 레이블을 렌더링합니다.

Returns:
            필요한 경우 왼쪽에서 잘린 텍스트 콘텐츠입니다.

        """
        width = self.content_size.width
        if not self.model or width <= 0:
            return ""
        full = f"{self.provider}:{self.model}" if self.provider else self.model
        if len(full) <= width:
            return Content(full)
        if len(self.model) <= width:
            return Content(self.model)
        if width > 1:
            return Content("\u2026" + self.model[-(width - 1) :])
        return Content("\u2026")


class StatusBar(Horizontal):
    """모드, 자동 승인, cwd, git 분기, 토큰, 모델을 표시하는 상태 표시줄."""

    DEFAULT_CSS = """
    StatusBar {
        height: 1;
        dock: bottom;
        background: $surface;
        padding: 0 1;
    }

    StatusBar .status-mode {
        width: auto;
        padding: 0 1;
    }

    StatusBar .status-mode.normal {
        display: none;
    }

    StatusBar .status-mode.shell {
        background: $mode-bash;
        color: white;
        text-style: bold;
    }

    StatusBar .status-mode.command {
        background: $mode-command;
        color: white;
    }

    StatusBar .status-auto-approve {
        width: auto;
        padding: 0 1;
    }

    StatusBar .status-auto-approve.on {
        background: $success;
        color: $background;
    }

    StatusBar .status-auto-approve.off {
        background: $warning;
        color: $background;
    }

    StatusBar .status-message {
        width: auto;
        padding: 0 1;
        color: $text-muted;
    }

    StatusBar .status-message.thinking {
        color: $warning;
    }

    StatusBar .status-cwd {
        width: auto;
        text-align: right;
        color: $text-muted;
    }

    StatusBar .status-branch {
        width: auto;
        color: $text-muted;
        padding: 0 1;
    }

    StatusBar .status-left-collapsible {
        width: 1fr;
        min-width: 0;
        height: 1;
        overflow-x: hidden;
    }

    StatusBar .status-tokens {
        width: auto;
        padding: 0 1;
        color: $text-muted;
    }

    StatusBar ModelLabel {
        width: auto;
        padding: 0 2;
        color: $text-muted;
        text-align: right;
    }
    """
    """모드 배지와 자동 승인 알약은 한눈에 상태를 확인할 수 있도록 고유한 색상을 사용합니다."""

    mode: reactive[str] = reactive("normal", init=False)
    status_message: reactive[str] = reactive("", init=False)
    auto_approve: reactive[bool] = reactive(default=False, init=False)
    cwd: reactive[str] = reactive("", init=False)
    branch: reactive[str] = reactive("", init=False)
    tokens: reactive[int] = reactive(0, init=False)

    def __init__(self, cwd: str | Path | None = None, **kwargs: Any) -> None:
        """상태 표시줄을 초기화합니다.

Args:
            cwd: 표시할 현재 작업 디렉터리
            **kwargs: 부모에게 전달된 추가 인수

        """
        super().__init__(**kwargs)
        # Store initial cwd - will be used in compose()
        self._initial_cwd = str(cwd) if cwd else str(Path.cwd())

    def compose(self) -> ComposeResult:  # noqa: PLR6301 — Textual widget method
        """상태 표시줄 레이아웃을 구성합니다.

Yields:
            모드, 자동 승인, 메시지, cwd, 지점, 토큰 등에 대한 위젯
                모델 디스플레이.

        """
        yield Static("", classes="status-mode normal", id="mode-indicator")
        yield Static(
            "manual | shift+tab to cycle",
            classes="status-auto-approve off",
            id="auto-approve-indicator",
        )
        with Horizontal(classes="status-left-collapsible"):
            yield Static("", classes="status-message", id="status-message")
            yield Static("", classes="status-cwd", id="cwd-display")
            yield Static("", classes="status-branch", id="branch-display")
        yield Static("", classes="status-tokens", id="tokens-display")
        yield ModelLabel(id="model-display")

    _BRANCH_WIDTH_THRESHOLD = 100
    """이 터미널 너비 아래에 git 분기 표시를 숨깁니다."""
    _CWD_WIDTH_THRESHOLD = 70
    """이 터미널 너비 아래에 cwd 표시를 숨깁니다."""

    def on_resize(self, event: events.Resize) -> None:
        """터미널 너비에 따라 상태 항목의 가시성을 관리합니다.

        우선순위(가장 높은 것부터): 모델, cwd, git 브랜치.

        """
        width = event.size.width
        with suppress(NoMatches):
            self.query_one("#branch-display", Static).display = (
                width >= self._BRANCH_WIDTH_THRESHOLD
            )
        with suppress(NoMatches):
            self.query_one("#cwd-display", Static).display = (
                width >= self._CWD_WIDTH_THRESHOLD
            )

    def on_mount(self) -> None:
        """감시자를 안전하게 트리거하려면 마운트 후 반응 값을 설정하세요."""
        from deepagents_cli.config import settings

        self.cwd = self._initial_cwd
        # Set initial model display
        label = self.query_one("#model-display", ModelLabel)
        label.provider = settings.model_provider or ""
        label.model = settings.model_name or ""

    def watch_mode(self, mode: str) -> None:
        """모드가 변경되면 모드 표시기를 업데이트합니다."""
        try:
            indicator = self.query_one("#mode-indicator", Static)
        except NoMatches:
            return
        indicator.remove_class("normal", "shell", "command")

        if mode == "shell":
            indicator.update("SHELL")
            indicator.add_class("shell")
        elif mode == "command":
            indicator.update("CMD")
            indicator.add_class("command")
        else:
            indicator.update("")
            indicator.add_class("normal")

    def watch_auto_approve(self, new_value: bool) -> None:
        """상태가 변경되면 자동 승인 표시기를 업데이트합니다."""
        try:
            indicator = self.query_one("#auto-approve-indicator", Static)
        except NoMatches:
            return
        indicator.remove_class("on", "off")

        if new_value:
            indicator.update("auto | shift+tab to cycle")
            indicator.add_class("on")
        else:
            indicator.update("manual | shift+tab to cycle")
            indicator.add_class("off")

    def watch_cwd(self, new_value: str) -> None:
        """cwd 표시가 변경되면 업데이트하세요."""
        try:
            display = self.query_one("#cwd-display", Static)
        except NoMatches:
            return
        display.update(self._format_cwd(new_value))

    def watch_branch(self, new_value: str) -> None:
        """변경 시 분기 표시를 업데이트합니다."""
        try:
            display = self.query_one("#branch-display", Static)
        except NoMatches:
            return
        icon = get_glyphs().git_branch
        display.update(f"{icon} {new_value}" if new_value else "")

    def watch_status_message(self, new_value: str) -> None:
        """상태 메시지 표시를 업데이트합니다."""
        try:
            msg_widget = self.query_one("#status-message", Static)
        except NoMatches:
            return

        msg_widget.remove_class("thinking")
        if new_value:
            msg_widget.update(new_value)
            if "thinking" in new_value.lower() or "executing" in new_value.lower():
                msg_widget.add_class("thinking")
        else:
            msg_widget.update("")

    def _format_cwd(self, cwd_path: str = "") -> str:
        """표시할 현재 작업 디렉터리의 형식을 지정합니다.

Returns:
            가능한 경우 홈 디렉터리에 ~를 사용하여 형식화된 경로 문자열입니다.

        """
        path = Path(cwd_path or self.cwd or self._initial_cwd)
        try:
            # Try to use ~ for home directory
            home = Path.home()
            if path.is_relative_to(home):
                return "~/" + path.relative_to(home).as_posix()
        except (ValueError, RuntimeError):
            pass
        return str(path)

    def set_mode(self, mode: str) -> None:
        """현재 입력 모드를 설정합니다.

Args:
            mode: "일반", "쉘", "명령" 중 하나

        """
        self.mode = mode

    def set_auto_approve(self, *, enabled: bool) -> None:
        """자동 승인 상태를 설정합니다.

Args:
            enabled: 자동 승인 활성화 여부

        """
        self.auto_approve = enabled

    def set_status_message(self, message: str) -> None:
        """상태 메시지를 설정합니다.

Args:
            message: 표시할 상태 메시지(지울 빈 문자열)

        """
        self.status_message = message

    _approximate: bool = False
    """표시된 값이 오래되었음을 알리려면 토큰 개수에 "+"를 추가하세요.

    (모델이 최종 사용량을 보고하기 전에 생성이 중단되었기 때문에 실제 컨텍스트는 더 큽니다.)

    """

    def watch_tokens(self, new_value: int) -> None:
        """개수가 변경되면 토큰 표시를 업데이트합니다."""
        self._render_tokens(new_value, approximate=self._approximate)

    def _render_tokens(self, count: int, *, approximate: bool = False) -> None:
        """토큰 수를 디스플레이 위젯에 렌더링합니다.

Args:
            count: 총 컨텍스트 토큰 수입니다.
            approximate: 개수가 오래되었음을 나타내려면 "+" 접미사를 추가합니다(예: 생성이 중단된 후).

        """
        try:
            display = self.query_one("#tokens-display", Static)
        except NoMatches:
            return

        if count > 0:
            suffix = "+" if approximate else ""
            # Format with K suffix for thousands
            if count >= 1000:  # noqa: PLR2004  # Count formatting threshold
                display.update(f"{count / 1000:.1f}K{suffix} tokens")
            else:
                display.update(f"{count}{suffix} tokens")
        else:
            display.update("")

    def set_tokens(self, count: int, *, approximate: bool = False) -> None:
        """토큰 수를 설정합니다.

        `hide_tokens`은 반응 속성을 업데이트하지 않고 위젯 텍스트를 지우기 때문에 값이 변경되지 않은 경우에도 디스플레이를 강제로 새로
        고칩니다.

Args:
            count: 현재 컨텍스트 토큰 수입니다.
            approximate: 개수가 오래되었음을 나타내려면 "+"를 추가합니다.

        """
        self._approximate = approximate
        if self.tokens == count:
            # Reactive dedup would skip the watcher — call render directly.
            self._render_tokens(count, approximate=approximate)
        else:
            # Reactive assignment triggers watch_tokens, which reads
            # self._approximate for the suffix.
            self.tokens = count

    def hide_tokens(self) -> None:
        """토큰 표시를 숨깁니다(예: 스트리밍 중)."""
        try:
            self.query_one("#tokens-display", Static).update("")
        except NoMatches:
            return

    def set_model(self, *, provider: str, model: str) -> None:
        """모델 표시 텍스트를 업데이트합니다.

Args:
            provider: 모델 제공자 이름(예: `'anthropic'`)
            model: 모델 이름(예: `'claude-sonnet-4-5'`).

        """
        label = self.query_one("#model-display", ModelLabel)
        label.provider = provider
        label.model = model
