"""에이전트 활동을 위한 작은 로딩 및 경과 시간 위젯.

이러한 위젯은 앱이 에이전트 출력 또는 백그라운드 작업이 완료되기를 기다리는 동안 사용되는 스피너 및 기간 표시를 제공합니다.
"""

from __future__ import annotations

from time import time
from typing import TYPE_CHECKING

from textual.containers import Horizontal
from textual.content import Content
from textual.widgets import Static

from deepagents_cli.config import get_glyphs
from deepagents_cli.formatting import format_duration

if TYPE_CHECKING:
    from textual.app import ComposeResult


class Spinner:
    """문자 세트에 적합한 프레임을 사용하는 애니메이션 스피너."""

    def __init__(self) -> None:
        """스피너를 초기화합니다."""
        self._position = 0

    @property
    def frames(self) -> tuple[str, ...]:
        """글리프 구성에서 스피너 프레임을 가져옵니다."""
        return get_glyphs().spinner_frames

    def next_frame(self) -> str:
        """다음 애니메이션 프레임을 가져옵니다.

Returns:
            애니메이션 시퀀스의 다음 스피너 캐릭터입니다.

        """
        frames = self.frames
        frame = frames[self._position]
        self._position = (self._position + 1) % len(frames)
        return frame

    def current_frame(self) -> str:
        """진행하지 않고 현재 프레임을 가져옵니다.

Returns:
            현재 스피너 캐릭터입니다.

        """
        return self.frames[self._position]


class LoadingWidget(Static):
    """상태 텍스트와 경과 시간이 포함된 애니메이션 로딩 표시기입니다.

    표시: <spinner> 생각 중...(3초, 중단하려면 esc)

    """

    DEFAULT_CSS = """
    LoadingWidget {
        height: auto;
        padding: 0 1;
        margin-top: 1;
    }

    LoadingWidget .loading-container {
        height: auto;
        width: 100%;
    }

    LoadingWidget .loading-spinner {
        width: auto;
        color: $primary;
    }

    LoadingWidget .loading-status {
        width: auto;
        color: $primary;
    }

    LoadingWidget .loading-hint {
        width: auto;
        color: $text-muted;
        margin-left: 1;
    }
    """

    def __init__(self, status: str = "Thinking") -> None:
        """로딩 위젯을 초기화합니다.

Args:
            status: 표시할 초기 상태 텍스트

        """
        super().__init__()
        self._status = status
        self._spinner = Spinner()
        self._start_time: float | None = None
        self._spinner_widget: Static | None = None
        self._status_widget: Static | None = None
        self._hint_widget: Static | None = None
        self._paused = False
        self._paused_elapsed: int = 0

    def compose(self) -> ComposeResult:
        """로딩 위젯 레이아웃을 구성합니다.

Yields:
            스피너, 상태 텍스트 및 힌트용 위젯입니다.

        """
        with Horizontal(classes="loading-container"):
            self._spinner_widget = Static(
                self._spinner.current_frame(), classes="loading-spinner"
            )
            yield self._spinner_widget

            self._status_widget = Static(
                f" {self._status}... ", classes="loading-status"
            )
            yield self._status_widget

            self._hint_widget = Static("(0s, esc to interrupt)", classes="loading-hint")
            yield self._hint_widget

    def on_mount(self) -> None:
        """마운트 시 애니메이션을 시작합니다."""
        self._start_time = time()
        self.set_interval(0.1, self._update_animation)

    def _update_animation(self) -> None:
        """스피너와 경과 시간을 업데이트합니다."""
        if self._paused:
            return

        if self._spinner_widget:
            frame = self._spinner.next_frame()
            self._spinner_widget.update(frame)

        if self._hint_widget and self._start_time is not None:
            elapsed = int(time() - self._start_time)
            self._hint_widget.update(f"({format_duration(elapsed)}, esc to interrupt)")

    def set_status(self, status: str) -> None:
        """상태 텍스트를 업데이트합니다.

Args:
            status: 새 상태 텍스트

        """
        self._status = status
        if self._status_widget:
            self._status_widget.update(f" {self._status}... ")

    def pause(self, status: str = "Awaiting decision") -> None:
        """애니메이션을 일시중지하고 상태를 업데이트합니다.

Args:
            status: 일시중지된 동안 표시되는 상태

        """
        self._paused = True
        if self._start_time is not None:
            self._paused_elapsed = int(time() - self._start_time)
        self._status = status
        if self._status_widget:
            self._status_widget.update(f" {status}... ")
        if self._hint_widget:
            self._hint_widget.update(
                f"(paused at {format_duration(self._paused_elapsed)})"
            )
        if self._spinner_widget:
            self._spinner_widget.update(Content.styled(get_glyphs().pause, "dim"))

    def resume(self) -> None:
        """애니메이션을 재개합니다."""
        self._paused = False
        self._status = "Thinking"
        if self._status_widget:
            self._status_widget.update(f" {self._status}... ")

    def stop(self) -> None:
        """애니메이션을 중지합니다(위젯은 호출자에 의해 제거됩니다)."""
