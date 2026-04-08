"""`deepagents-cli`을 지원하는 텍스트 애플리케이션을 실행하세요.

앱은 장기 실행 백그라운드 작업 중에 터미널 UI의 응답성을 유지하면서 위젯 상태, 에이전트 실행, 스레드 수명 주기, 도구 승인 흐름 및 사용자 대상 명령을
조정합니다.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import signal
import sys
import time
import uuid
import webbrowser
from collections import deque
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from textual.app import App, ScreenStackError
from textual.binding import Binding, BindingType
from textual.containers import Container, VerticalScroll
from textual.content import Content
from textual.css.query import NoMatches
from textual.message import Message
from textual.screen import ModalScreen
from textual.style import Style as TStyle
from textual.theme import Theme
from textual.widgets import Static

from deepagents_cli import theme
from deepagents_cli._cli_context import CLIContext
from deepagents_cli._session_stats import (
    SessionStats,
    SpinnerStatus,
    format_token_count,
)

# Only is_ascii_mode is needed before first paint (on_mount scrollbar config).
# All other config imports — settings, create_model, detect_provider, etc. — are
# deferred to local imports at their call sites since they are only accessed
# after user interaction begins.
from deepagents_cli._version import CHANGELOG_URL, DOCS_URL
from deepagents_cli.config import is_ascii_mode
from deepagents_cli.widgets.chat_input import ChatInput
from deepagents_cli.widgets.loading import LoadingWidget
from deepagents_cli.widgets.message_store import (
    MessageData,
    MessageStore,
    MessageType,
    ToolStatus,
)
from deepagents_cli.widgets.messages import (
    AppMessage,
    AssistantMessage,
    ErrorMessage,
    QueuedUserMessage,
    SkillMessage,
    ToolCallMessage,
    UserMessage,
)
from deepagents_cli.widgets.status import StatusBar
from deepagents_cli.widgets.welcome import WelcomeBanner

logger = logging.getLogger(__name__)
_monotonic = time.monotonic

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from deepagents.backends import CompositeBackend
    from langchain_core.runnables import RunnableConfig
    from langgraph.pregel import Pregel
    from textual.app import ComposeResult
    from textual.events import Click, MouseUp, Paste
    from textual.scrollbar import ScrollUp
    from textual.widget import Widget
    from textual.worker import Worker

    from deepagents_cli._ask_user_types import AskUserWidgetResult, Question
    from deepagents_cli.mcp_tools import MCPServerInfo
    from deepagents_cli.remote_client import RemoteAgent
    from deepagents_cli.server import ServerProcess
    from deepagents_cli.skills.load import ExtendedSkillMetadata
    from deepagents_cli.textual_adapter import TextualUIAdapter
    from deepagents_cli.widgets.approval import ApprovalMenu
    from deepagents_cli.widgets.ask_user import AskUserMenu

# ---------------------------------------------------------------------------
# Terminal-specific startup workarounds
# ---------------------------------------------------------------------------
# iTerm2's cursor guide (highlight cursor line) causes visual artifacts when
# Textual takes over the terminal in alternate screen mode. We disable it at
# module load and restore on exit. Both atexit and exit() override are used
# for defense-in-depth: atexit catches abnormal termination (SIGTERM, unhandled
# exceptions), while exit() ensures restoration before Textual's cleanup.

# Detection: check env vars AND that stderr is a TTY (avoids false positives
# when env vars are inherited but running in non-TTY context like CI)
_IS_ITERM = (
    (
        os.environ.get("LC_TERMINAL", "") == "iTerm2"
        or os.environ.get("TERM_PROGRAM", "") == "iTerm.app"
    )
    and hasattr(os, "isatty")
    and os.isatty(2)
)

# iTerm2 cursor guide escape sequences (OSC 1337)
# Format: OSC 1337 ; HighlightCursorLine=<yes|no> ST
# Where OSC = ESC ] (0x1b 0x5d) and ST = ESC \ (0x1b 0x5c)
_ITERM_CURSOR_GUIDE_OFF = "\x1b]1337;HighlightCursorLine=no\x1b\\"
_ITERM_CURSOR_GUIDE_ON = "\x1b]1337;HighlightCursorLine=yes\x1b\\"


def _write_iterm_escape(sequence: str) -> None:
    """iTerm2 이스케이프 시퀀스를 stderr에 작성합니다.

    터미널을 사용할 수 없는 경우(리디렉션, 폐쇄, 파이프 파손) 자동으로 실패합니다. 이는 외관상의 기능이므로 오류로 인해 앱이 충돌해서는 안 됩니다.

    """
    if not _IS_ITERM:
        return
    try:
        import sys

        if sys.__stderr__ is not None:
            sys.__stderr__.write(sequence)
            sys.__stderr__.flush()
    except OSError:
        # Terminal may be unavailable (redirected, closed, broken pipe)
        pass


# Disable cursor guide at module load (before Textual takes over)
_write_iterm_escape(_ITERM_CURSOR_GUIDE_OFF)

if _IS_ITERM:
    import atexit

    def _restore_cursor_guide() -> None:
        """종료 시 iTerm2 커서 안내를 복원합니다.

        종료 발생 방식에 관계없이 CLI가 종료될 때 커서 안내가 다시 활성화되도록 atexit에 등록되었습니다.

        """
        _write_iterm_escape(_ITERM_CURSOR_GUIDE_ON)

    atexit.register(_restore_cursor_guide)


# ---------------------------------------------------------------------------
# Theme persistence and startup argument parsing helpers
# ---------------------------------------------------------------------------


def _load_theme_preference() -> str:
    """구성에서 저장된 테마 이름을 로드하거나 기본값을 반환합니다.

Returns:
        텍스트 테마 이름(예: `'langchain'`, `'langchain-light'`).

    """
    import tomllib

    try:
        from deepagents_cli.model_config import DEFAULT_CONFIG_PATH

        if not DEFAULT_CONFIG_PATH.exists():
            return theme.DEFAULT_THEME

        with DEFAULT_CONFIG_PATH.open("rb") as f:
            data = tomllib.load(f)
    except (tomllib.TOMLDecodeError, PermissionError, OSError) as exc:
        logger.warning("Could not read config for theme preference: %s", exc)
        return theme.DEFAULT_THEME

    name = data.get("ui", {}).get("theme")
    if isinstance(name, str) and name in theme.ThemeEntry.REGISTRY:
        return name
    if isinstance(name, str):
        logger.warning(
            "Unknown theme '%s' in config; falling back to default",
            name,
        )
    return theme.DEFAULT_THEME


def save_theme_preference(name: str) -> bool:
    """`~/.deepagents/config.toml`에 대한 테마 기본 설정을 유지합니다.

Args:
        name: 저장할 텍스트 테마 이름입니다.

Returns:
        기본 설정이 저장된 경우 `True`, 오류가 발생한 경우 `False`입니다.

    """
    if name not in theme.ThemeEntry.REGISTRY:
        logger.warning("Refusing to save unknown theme '%s'", name)
        return False

    import contextlib
    import tempfile

    try:
        import tomllib

        import tomli_w

        from deepagents_cli.model_config import DEFAULT_CONFIG_PATH

        DEFAULT_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        if DEFAULT_CONFIG_PATH.exists():
            with DEFAULT_CONFIG_PATH.open("rb") as f:
                data = tomllib.load(f)
        else:
            data = {}

        if "ui" not in data:
            data["ui"] = {}
        data["ui"]["theme"] = name

        fd, tmp_path = tempfile.mkstemp(dir=DEFAULT_CONFIG_PATH.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                tomli_w.dump(data, f)
            Path(tmp_path).replace(DEFAULT_CONFIG_PATH)
        except BaseException:
            with contextlib.suppress(OSError):
                Path(tmp_path).unlink()
            raise
    except Exception:
        logger.exception("Could not save theme preference")
        return False
    return True


def _extract_model_params_flag(raw_arg: str) -> tuple[str, dict[str, Any] | None]:
    """`/model` 인수 문자열에서 `--model-params` 및 해당 JSON 값을 추출합니다.

    공백이 포함된 JSON이 인용 없이 작동하도록 균형 잡힌 중괄호로 인용된 값(`'...'` / `"..."`)과 순수 `{...}` 값을 처리합니다.

Note:
        bare-brace 모드는 JSON 문자열 내용을 인식하지 못한 채 `{` / `}` 문자를 계산합니다. 문자열 안에 리터럴 중괄호가 포함된
        값(예: `{"stop": "end}here"}`)은 잘못 구문 분석됩니다. 이 경우 사용자는 해당 값을 인용해야 합니다.

Args:
        raw_arg: `/model ` 뒤의 인수 문자열입니다.

Returns:
        `(remaining_args, parsed_dict | None)`의 튜플입니다. 다음에 대해 `None`을(를) 반환합니다.
            플래그가 없을 때 dict.

Raises:
        ValueError: 값이 누락되었거나, 닫히지 않은 따옴표, 불균형 중괄호가 있거나 유효한 JSON이 아닌 경우입니다.
        TypeError: 구문 분석된 JSON이 dict가 아닌 경우.

    """
    flag = "--model-params"
    idx = raw_arg.find(flag)
    if idx == -1:
        return raw_arg, None

    before = raw_arg[:idx].rstrip()
    after = raw_arg[idx + len(flag) :].lstrip()

    if not after:
        msg = "--model-params requires a JSON object value"
        raise ValueError(msg)

    # Determine the JSON string boundaries.
    if after[0] in {"'", '"'}:
        quote = after[0]
        end = -1
        backslash_count = 0
        for i, ch in enumerate(after[1:], start=1):
            if ch == "\\":
                backslash_count += 1
                continue
            if ch == quote and backslash_count % 2 == 0:
                end = i
                break
            backslash_count = 0
        if end == -1:
            msg = f"Unclosed {quote} in --model-params value"
            raise ValueError(msg)
        # Parse the quoted token with shlex so escaped quotes are unescaped.
        json_str = shlex.split(after[: end + 1], posix=True)[0]
        rest = after[end + 1 :].lstrip()
    elif after[0] == "{":
        # Walk forward to find the matching closing brace.
        depth = 0
        end = -1
        for i, ch in enumerate(after):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end == -1:
            msg = "Unbalanced braces in --model-params value"
            raise ValueError(msg)
        json_str = after[: end + 1]
        rest = after[end + 1 :].lstrip()
    else:
        # Non-brace, non-quoted — take the next whitespace-delimited token.
        parts = after.split(None, 1)
        json_str = parts[0]
        rest = parts[1] if len(parts) > 1 else ""

    remaining = f"{before} {rest}".strip()
    try:
        params = json.loads(json_str)
    except json.JSONDecodeError:
        msg = (
            f"Invalid JSON in --model-params: {json_str!r}. "
            'Expected format: --model-params \'{"key": "value"}\''
        )
        raise ValueError(msg) from None
    if not isinstance(params, dict):
        msg = "--model-params must be a JSON object, got " + type(params).__name__
        raise TypeError(msg)
    return remaining, params


InputMode = Literal["normal", "shell", "command"]

_TYPING_IDLE_THRESHOLD_SECONDS: float = 2.0
"""마지막 키 입력 이후 사용자가 유휴 상태로 간주되고 보류 중인 승인 위젯이 표시될 수 있는 시간(초)입니다.

2초는 실수로 승인 키를 누르는 것을 방지하는 동시에 응답 속도의 균형을 유지합니다.
"""

_DEFERRED_APPROVAL_TIMEOUT_SECONDS: float = 30.0
"""승인 지연 작업자가 승인 위젯을 표시하기 전에 사용자가 입력을 멈출 때까지 기다리는 최대 시간(초)입니다."""


@dataclass(frozen=True, slots=True)
class QueuedMessage:
    """처리를 기다리는 대기 중인 사용자 메시지를 나타냅니다."""

    text: str
    """메시지 텍스트 내용입니다."""

    mode: InputMode
    """메시지 라우팅을 결정하는 입력 모드입니다."""


DeferredActionKind = Literal["model_switch", "thread_switch", "chat_output"]
"""유형 확인 중복 제거에 유효한 `DeferredAction.kind` 값입니다."""


@dataclass(frozen=True, slots=True, kw_only=True)
class DeferredAction:
    """현재 사용 중 상태가 해결될 때까지 연기되는 작업입니다."""

    kind: DeferredActionKind
    """중복 제거를 위한 ID 키 — `DeferredActionKind` 중 하나입니다."""

    execute: Callable[[], Awaitable[None]]
    """실제 작업을 수행하는 비동기 호출 가능 항목입니다."""


@dataclass(frozen=True, slots=True)
class _ThreadHistoryPayload:
    """`_fetch_thread_history_data`에서 반환된 데이터입니다."""

    messages: list[MessageData]
    """변환된 메시지 데이터를 대량 로드할 준비가 되었습니다."""

    context_tokens: int
    """체크포인트에서 `_context_tokens`을 유지했습니다(없는 경우 0)."""


def _new_thread_id() -> str:
    """`sessions.generate_thread_id` 주변의 지연된 가져오기 래퍼입니다.

Returns:
        UUID7 문자열.

    """
    from deepagents_cli.sessions import generate_thread_id

    return generate_thread_id()


# ---------------------------------------------------------------------------
# Mutable session state and the main Textual application
# ---------------------------------------------------------------------------


class TextualSessionState:
    """Textual 앱의 세션 상태입니다."""

    def __init__(
        self,
        *,
        auto_approve: bool = False,
        thread_id: str | None = None,
    ) -> None:
        """세션 상태를 초기화합니다.

Args:
            auto_approve: 도구 호출 자동 승인 여부
            thread_id: 선택적 스레드 ID(제공되지 않은 경우 UUID7 생성)

        """
        self.auto_approve = auto_approve
        self.thread_id = thread_id or _new_thread_id()

    def reset_thread(self) -> str:
        """새 스레드로 재설정합니다.

Returns:
            새 thread_id입니다.

        """
        self.thread_id = _new_thread_id()
        return self.thread_id


_COMMAND_URLS: dict[str, str] = {
    "/changelog": CHANGELOG_URL,
    "/docs": DOCS_URL,
    "/feedback": "https://github.com/langchain-ai/deepagents/issues/new/choose",
}
"""브라우저를 여는 명령에 대한 URL 매핑에 대한 슬래시 명령입니다."""


class DeepAgentsApp(App):
    """deepagents-cli에 대한 기본 텍스트 응용 프로그램입니다."""

    TITLE = "Deep Agents"
    """텍스트 응용 프로그램 제목입니다."""

    CSS_PATH = "app.tcss"
    """앱 레이아웃의 텍스트 CSS 스타일시트 경로입니다."""

    ENABLE_COMMAND_PALETTE = False
    """사용자 정의 슬래시를 위해 Textual의 내장 명령 팔레트를 비활성화합니다.
    명령 시스템.
    """

    SCROLL_SENSITIVITY_Y = 1.0
    """수직 스크롤 속도(세밀한 제어를 위해 텍스트 기본값에서 감소)"""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "interrupt", "Interrupt", show=False, priority=True),
        Binding(
            "ctrl+c",
            "quit_or_interrupt",
            "Quit/Interrupt",
            show=False,
            priority=True,
        ),
        Binding("ctrl+d", "quit_app", "Quit", show=False, priority=True),
        Binding("ctrl+t", "toggle_auto_approve", "Toggle Auto-Approve", show=False),
        Binding(
            "shift+tab",
            "toggle_auto_approve",
            "Toggle Auto-Approve",
            show=False,
            priority=True,
        ),
        Binding(
            "ctrl+o",
            "toggle_tool_output",
            "Toggle Tool Output",
            show=False,
            priority=True,
        ),
        Binding(
            "ctrl+x",
            "open_editor",
            "Open Editor",
            show=False,
            priority=True,
        ),
        # Approval menu keys (handled at App level for reliability)
        Binding("up", "approval_up", "Up", show=False),
        Binding("k", "approval_up", "Up", show=False),
        Binding("down", "approval_down", "Down", show=False),
        Binding("j", "approval_down", "Down", show=False),
        Binding("enter", "approval_select", "Select", show=False),
        Binding("y", "approval_yes", "Yes", show=False),
        Binding("1", "approval_yes", "Yes", show=False),
        Binding("2", "approval_auto", "Auto", show=False),
        Binding("a", "approval_auto", "Auto", show=False),
        Binding("3", "approval_no", "No", show=False),
        Binding("n", "approval_no", "No", show=False),
    ]
    """중단, 종료, 토글 및 승인 메뉴에 대한 앱 수준 키 바인딩
    항해.
    """

    class ServerReady(Message):
        """백그라운드 서버 시작 작업자가 성공 시 게시한 내용입니다."""

        def __init__(  # noqa: D107
            self,
            agent: Any,  # noqa: ANN401
            server_proc: Any,  # noqa: ANN401
            mcp_server_info: list[Any] | None,
        ) -> None:
            super().__init__()
            self.agent = agent
            self.server_proc = server_proc
            self.mcp_server_info = mcp_server_info

    class ServerStartFailed(Message):
        """실패 시 백그라운드 서버 시작 작업자가 게시합니다."""

        def __init__(self, error: Exception) -> None:  # noqa: D107
            super().__init__()
            self.error = error

    def __init__(
        self,
        *,
        agent: Pregel | None = None,
        assistant_id: str | None = None,
        backend: CompositeBackend | None = None,
        auto_approve: bool = False,
        cwd: str | Path | None = None,
        thread_id: str | None = None,
        resume_thread: str | None = None,
        initial_prompt: str | None = None,
        mcp_server_info: list[MCPServerInfo] | None = None,
        profile_override: dict[str, Any] | None = None,
        server_proc: ServerProcess | None = None,
        server_kwargs: dict[str, Any] | None = None,
        mcp_preload_kwargs: dict[str, Any] | None = None,
        model_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Deep Agents 애플리케이션을 초기화합니다.

Args:
            agent: 사전 구성된 LangGraph 에이전트 또는 `None`(서버 시작이 `server_kwargs`을 통해 지연되는 경우).
            assistant_id: 메모리 저장을 위한 에이전트 식별자
            backend: 파일 작업을 위한 백엔드
            auto_approve: 자동 승인을 활성화한 상태로 시작할지 여부
            cwd: 표시할 현재 작업 디렉터리
            thread_id: 세션의 스레드 ID입니다.

                `resume_thread`이 제공되는 경우 `None`(비동기적으로 해결됨)
            resume_thread: `-r` 플래그의 원시 재개 의도입니다.

                `-r`의 경우 `'__MOST_RECENT__'`, `-r <id>`의 경우 스레드 ID 문자열, 새 세션의 경우
                `None`입니다.

                `_start_server_background` 동안 `_resolve_resume_thread`을(를) 통해 해결되었습니다.

                `server_kwargs`을 설정해야 합니다. 그렇지 않으면 무시됩니다.
            initial_prompt: 세션이 시작될 때 자동 제출하라는 선택적 프롬프트
            mcp_server_info: `/mcp` 뷰어의 MCP 서버 메타데이터입니다.
            profile_override: `--profile-override`의 추가 프로필 필드는 모델 선택 세부 정보, 오프로드 예산 표시,
                              `/offload`와 같은 주문형 `create_model()` 호출을 포함하여 나중에 프로필 인식
                              동작이 CLI 재정의와 일관되게 유지되도록 유지됩니다.
            server_proc: 대화형 세션을 위한 LangGraph 서버 프로세스입니다.
            server_kwargs: 제공되면 서버 시작이 지연됩니다.

                앱에 "연결 중..." 상태가 표시되고 `start_server_and_get_agent`에 대해 이러한 kwargs를 사용하여
                백그라운드에서 서버를 시작합니다.
            mcp_preload_kwargs: `_preload_session_mcp_server_info`용 Kwargs는
                                `server_kwargs`이 설정되면 서버 시작과 동시에 실행됩니다.
            model_kwargs: 지연된 `create_model()`에 대한 Kwargs.

                제공되면 시작을 차단하는 대신 첫 번째 페인트 후 백그라운드 작업자에서 모델 생성이 실행됩니다.
            **kwargs: 부모에게 전달된 추가 인수

        """
        super().__init__(**kwargs)

        self._register_custom_themes()

        # Apply saved theme preference (or default)
        self.theme = _load_theme_preference()

        self._agent = agent

        self._assistant_id = assistant_id

        self._backend = backend

        self._auto_approve = auto_approve

        self._cwd = str(cwd) if cwd else str(Path.cwd())

        self._lc_thread_id = thread_id
        """LangChain 스레드 식별자입니다.

        Textual의 `App._thread_id`과의 충돌을 피하기 위해 `_lc_thread_id`로 명명되었습니다.

        """

        self._resume_thread_intent = resume_thread

        self._initial_prompt = initial_prompt

        self._mcp_server_info = mcp_server_info

        self._profile_override = profile_override

        self._server_proc = server_proc

        self._server_kwargs = server_kwargs

        self._mcp_preload_kwargs = mcp_preload_kwargs

        self._model_kwargs = model_kwargs

        self._connecting = server_kwargs is not None
        # Extract sandbox type from server kwargs for trace metadata.
        # ServerConfig.__post_init__ normalizes "none" → None, but server_kwargs carries
        # the raw argparse value, so guard against both.

        raw = (server_kwargs or {}).get("sandbox_type")

        self._sandbox_type: str | None = raw if raw and raw != "none" else None

        self._model_override: str | None = None

        self._model_params_override: dict[str, Any] | None = None

        self._mcp_tool_count = sum(len(s.tools) for s in (mcp_server_info or []))

        self._status_bar: StatusBar | None = None

        self._chat_input: ChatInput | None = None

        self._quit_pending = False

        self._session_state: TextualSessionState | None = None

        self._ui_adapter: TextualUIAdapter | None = None

        self._pending_approval_widget: ApprovalMenu | None = None

        self._pending_ask_user_widget: AskUserMenu | None = None
        # Agent task tracking for interruption

        self._agent_worker: Worker[None] | None = None

        self._agent_running = False

        self._shell_process: asyncio.subprocess.Process | None = None
        """중단에 대한 쉘 명령 프로세스 추적(! 명령)."""

        self._shell_worker: Worker[None] | None = None

        self._shell_running = False

        self._loading_widget: LoadingWidget | None = None

        self._context_tokens: int = 0
        """마지막 전체 컨텍스트 토큰 수의 로컬 캐시입니다.

        정보의 출처는 그래프 상태의 `_context_tokens`입니다. 이는 상태 표시줄의 동기화 복사본입니다.

        """

        self._tokens_approximate: bool = False
        """캐시된 토큰 수가 오래되었는지 여부(중단된 생성)"""

        self._last_typed_at: float | None = None
        """입력 인식 승인 연기 상태입니다."""

        self._approval_placeholder: Static | None = None

        self._update_available: tuple[bool, str | None] = (False, None)
        """가용성 상태 업데이트 — `_check_for_updates`에 의해 설정되며 종료 시 읽습니다."""

        self._session_stats: SessionStats = SessionStats()
        """이 세션의 모든 턴에 대한 누적 사용 통계입니다."""

        self._inflight_turn_stats: SessionStats | None = None
        """현재 실행 중인 턴에 대한 통계입니다.

        이벤트 루프가 종료되기 전에 `exit()`이 이를 동기식으로 병합할 수 있도록 여기에 보관됩니다(예: 보류 중인 도구 호출 중
        `Ctrl+D`).

        """

        self._inflight_turn_start: float = 0.0
        """현재 턴이 시작되었을 때의 단조로운 타임스탬프입니다."""

        self._pending_messages: deque[QueuedMessage] = deque()
        """순차적 처리를 위한 사용자 메시지 큐입니다."""

        self._queued_widgets: deque[QueuedUserMessage] = deque()

        self._processing_pending = False

        self._thread_switching = False

        self._model_switching = False

        self._deferred_actions: list[DeferredAction] = []
        """현재 사용 중 상태가 해결된 후 실행되는 지연된 작업입니다."""

        self._message_store = MessageStore()
        """메시지 가상화 저장소."""

        self._startup_task: asyncio.Task[None] | None = None
        """시작 작업 참조(on_mount에 설정)"""

        self._discovered_skills: list[ExtendedSkillMetadata] = []
        """캐시된 기술 메타데이터(시작 검색 작업자에 의해 채워짐,
        `/reload`에 새로 고쳐졌습니다).

        `_handle_skill_command`에서 모든 호출 시 모든 스킬 디렉터리 재탐색을 건너뛰는 데 사용됩니다.

        """

        self._skill_allowed_roots: list[Path] = []
        """격리 확인을 위해 사전 해결된 스킬 루트 디렉터리
        `load_skill_content`.

        `_discovered_skills`과 함께 구축되었습니다.

        """

        # Lazily imported here to avoid pulling image dependencies into
        # argument parsing paths.
        from deepagents_cli.input import MediaTracker

        self._image_tracker = MediaTracker()

    def _remote_agent(self) -> RemoteAgent | None:
        """`RemoteAgent` 또는 `None`로 축소된 에이전트를 반환합니다.

        다음과 같은 경우 `None`을 반환합니다.

        - 구성된 에이전트가 없습니다(`self._agent is None`). - 에이전트는 로컬 `Pregel` 그래프입니다(예: ACP 모드,
        테스트 하네스).

        서버 지원 에이전트가 필요한 기능을 제어하는 ​​데 사용됩니다(예: `ConfigurableModelMiddleware`을 통한 모델 전환,
        체크포인터 대체). 서버 소유권이 아닌 에이전트 유형을 확인하므로 이는 CLI 생성 서버와 외부에서 관리되는 서버 모두에 대해 작동합니다.

Returns:
            `RemoteAgent` 인스턴스 또는 로컬 에이전트의 경우 `None`입니다.

        """
        from deepagents_cli.remote_client import RemoteAgent

        return self._agent if isinstance(self._agent, RemoteAgent) else None

    def get_theme_variable_defaults(self) -> dict[str, str]:
        """현재 테마에 대한 사용자 정의 CSS 변수 기본값을 반환합니다.

        대부분의 스타일은 Textual의 내장 변수(`$primary`, `$text-muted`, `$error-muted` 등)를 사용합니다.  이
        재정의는 해당 텍스트가 없는 앱별 변수(`$mode-bash`, `$mode-command`, `$skill`, `$skill-hover`,
        `$tool`, `$tool-hover`)를 삽입합니다.

Returns:
            16진수 색상 값에 대한 CSS 변수 이름의 사전입니다.

        """
        colors = theme.get_theme_colors(self)
        return theme.get_css_variable_defaults(colors=colors)

    def compose(self) -> ComposeResult:
        """애플리케이션 레이아웃을 구성합니다.

Yields:
            기본 채팅 영역 및 상태 표시줄의 UI 구성요소입니다.

        """
        # Main chat area with scrollable messages
        # VerticalScroll tracks user scroll intent for better auto-scroll behavior
        with VerticalScroll(id="chat"):
            yield WelcomeBanner(
                thread_id=self._lc_thread_id,
                mcp_tool_count=self._mcp_tool_count,
                connecting=self._connecting,
                resuming=self._resume_thread_intent is not None,
                local_server=self._server_kwargs is not None,
                id="welcome-banner",
            )
            yield Container(id="messages")
        with Container(id="bottom-app-container"):
            yield ChatInput(
                cwd=self._cwd,
                image_tracker=self._image_tracker,
                id="input-area",
            )

        # Status bar at bottom
        yield StatusBar(cwd=self._cwd, id="status-bar")

    async def on_mount(self) -> None:
        """마운트 후 구성요소를 초기화합니다.

        여기에는 위젯 쿼리와 경량 구성만 있습니다. 첫 번째 렌더링된 프레임을 지연시키는 모든 것(하위 프로세스 호출, 과도한 가져오기)은
        `call_after_refresh`을 통해 `_post_paint_init`로 연기됩니다.

        """
        # Move all objects allocated during import/compose into the permanent
        # generation so the cyclic GC skips them during first-paint rendering.
        import gc

        gc.freeze()

        chat = self.query_one("#chat", VerticalScroll)
        chat.anchor()
        if is_ascii_mode():
            chat.styles.scrollbar_size_vertical = 0

        self._status_bar = self.query_one("#status-bar", StatusBar)
        self._chat_input = self.query_one("#input-area", ChatInput)

        # Apply any skill commands discovered before the widget was mounted
        if self._discovered_skills:
            from deepagents_cli.command_registry import (
                SLASH_COMMANDS,
                build_skill_commands,
            )

            cmds = build_skill_commands(self._discovered_skills)
            merged = list(SLASH_COMMANDS) + cmds
            self._chat_input.update_slash_commands(merged)

        # Set initial auto-approve state
        if self._auto_approve:
            self._status_bar.set_auto_approve(enabled=True)

        # Focus the input immediately so the cursor is visible on first paint
        self._chat_input.focus_input()

        # Prewarm heavy imports in a thread while the first frame renders.
        # The user can't type yet, so GIL contention is harmless.  By the
        # time _post_paint_init fires its inline imports are dict lookups.
        self.run_worker(
            asyncio.to_thread(self._prewarm_deferred_imports),
            exclusive=True,
            group="startup-import-prewarm",
        )

        # Start branch resolution immediately — the thread launches now
        # (during on_mount) so by the time the first frame finishes painting
        # the subprocess is already done. _post_paint_init fires the heavier
        # workers (server, model creation) afterward.
        self._startup_task = asyncio.create_task(
            self._resolve_git_branch_and_continue()
        )

    async def _resolve_git_branch_and_continue(self) -> None:
        """git 분기를 해결한 다음 나머지 초기화 작업자를 예약합니다.

        `on_mount` 중에 `asyncio.create_task()`을 통해 시작되므로 하위 프로세스가 첫 번째 페인트 렌더링과 동시에
        실행됩니다. `_post_paint_init`은 분기 확인 성공 여부에 관계없이 `call_after_refresh`을 통해 예약됩니다.

        """
        try:
            import subprocess  # noqa: S404  # stdlib, already loaded

            def _get_branch() -> str:
                try:
                    result = subprocess.run(
                        ["git", "rev-parse", "--abbrev-ref", "HEAD"],  # noqa: S607
                        capture_output=True,
                        text=True,
                        timeout=2,
                        check=False,
                    )
                    if result.returncode == 0:
                        return result.stdout.strip()
                except FileNotFoundError:
                    pass  # git not installed
                except subprocess.TimeoutExpired:
                    logger.debug("Git branch detection timed out")
                except OSError:
                    logger.debug("Git branch detection failed", exc_info=True)
                return ""

            branch = await asyncio.to_thread(_get_branch)
            if self._status_bar:
                self._status_bar.branch = branch
        except Exception:
            logger.warning("Git branch resolution failed", exc_info=True)
        finally:
            # Always schedule post-paint init — even if branch resolution
            # fails, the app must still start the server, session, etc.
            self.call_after_refresh(self._post_paint_init)

    async def _post_paint_init(self) -> None:
        """남은 시작 작업을 위해 백그라운드 작업자를 해고합니다.

        여기에 있는 모든 것은 비차단입니다. 즉, 작업자 및 스레드 오프로드 호출을 통해 UI가 응답성을 유지합니다.

        """
        # Create UI adapter unconditionally — it only holds UI callbacks and
        # doesn't depend on the agent. The agent is injected later at
        # execute_task_textual() call time.
        from deepagents_cli.textual_adapter import TextualUIAdapter

        self._ui_adapter = TextualUIAdapter(
            mount_message=self._mount_message,
            update_status=self._update_status,
            request_approval=self._request_approval,
            on_auto_approve_enabled=self._on_auto_approve_enabled,
            set_spinner=self._set_spinner,
            set_active_message=self._set_active_message,
            sync_message_content=self._sync_message_content,
            request_ask_user=self._request_ask_user,
        )
        # Wire token display callbacks
        self._ui_adapter._on_tokens_update = self._on_tokens_update
        self._ui_adapter._on_tokens_hide = self._hide_tokens
        self._ui_adapter._on_tokens_show = self._show_tokens

        # Fire-and-forget workers — none of these block the event loop.

        # Discover skills first so /skill: autocomplete is ready as early
        # as possible. The heavy filesystem scan runs in a thread.
        self.run_worker(
            self._discover_skills(),
            exclusive=True,
            group="startup-skill-discovery",
        )

        self.run_worker(self._init_session_state, exclusive=True, group="session-init")

        # Server startup (model creation + server process)
        if self._server_kwargs is not None:
            self.run_worker(
                self._start_server_background,
                exclusive=True,
                group="server-startup",
            )

        # Background update check and what's-new banner
        # (opt-out via env var or config.toml [update].check)
        from deepagents_cli.update_check import is_update_check_enabled

        if is_update_check_enabled():
            self.run_worker(
                self._check_for_updates,
                exclusive=True,
                group="startup-update-check",
            )
            self.run_worker(
                self._show_whats_new,
                exclusive=True,
                group="startup-whats-new",
            )

        # Prewarm model discovery and profile caches unconditionally so
        # /model opens instantly even before the agent/server is ready.
        self.run_worker(
            self._prewarm_model_caches,
            exclusive=True,
            group="startup-model-prewarm",
        )

        # Prewarm thread message counts so /threads opens instantly.
        self.run_worker(
            self._prewarm_threads_cache,
            exclusive=True,
            group="startup-thread-prewarm",
        )

        # Optional tool warnings in a thread (shutil.which is sync I/O)
        self.run_worker(
            self._check_optional_tools_background,
            exclusive=True,
            group="startup-tool-check",
        )

        # Auto-submit initial prompt if provided via -m flag.
        # This check must come first because _lc_thread_id and _agent are
        # always set (even for brand-new sessions), so an elif after the
        # thread-history branch would never execute.
        # When connecting, defer until on_deep_agents_app_server_ready fires.
        if not self._connecting:
            if self._initial_prompt and self._initial_prompt.strip():
                prompt = self._initial_prompt
                self.call_after_refresh(
                    lambda: asyncio.create_task(self._handle_user_message(prompt))
                )
            elif self._lc_thread_id and self._agent:
                self.call_after_refresh(
                    lambda: asyncio.create_task(self._load_thread_history())
                )

    async def _init_session_state(self) -> None:
        """스레드에서 세션 상태를 생성합니다(deepagents_cli.sessions 가져오기)."""

        def _create() -> TextualSessionState:
            return TextualSessionState(
                auto_approve=self._auto_approve,
                thread_id=self._lc_thread_id,
            )

        try:
            self._session_state = await asyncio.to_thread(_create)
        except Exception:
            logger.exception("Failed to create session state")
            self.notify(
                "Session initialization failed. Some features may be unavailable.",
                severity="error",
                timeout=10,
            )

    async def _check_optional_tools_background(self) -> None:
        """스레드에서 선택적 도구를 확인하고 누락된 경우 알림을 보냅니다."""
        try:
            from deepagents_cli.main import (
                check_optional_tools,
                format_tool_warning_tui,
            )
        except ImportError:
            logger.warning(
                "Could not import optional tools checker",
                exc_info=True,
            )
            return

        try:
            missing = await asyncio.to_thread(check_optional_tools)
        except (OSError, FileNotFoundError):
            logger.debug("Failed to check for optional tools", exc_info=True)
            return
        except Exception:
            logger.warning("Unexpected error checking optional tools", exc_info=True)
            return

        for tool in missing:
            self.notify(
                format_tool_warning_tui(tool),
                severity="warning",
                timeout=15,
                markup=False,
            )

    async def _discover_skills(self) -> None:
        """기술을 발견하고, 메타데이터를 캐시하고, 자동 완성을 업데이트하세요.

        `/skill:<name>` 호출이 모든 기술 디렉터리 재탐색을 건너뛸 수 있도록 전체 `ExtendedSkillMetadata` 목록과 사전
        해결된 포함 루트를 캐시합니다.

        이벤트 루프 차단을 방지하기 위해 스레드에서 파일 시스템 I/O를 실행합니다.

        """
        from deepagents_cli.command_registry import SLASH_COMMANDS, build_skill_commands

        try:
            skills, roots = await asyncio.to_thread(self._discover_skills_and_roots)
            self._discovered_skills = skills
            self._skill_allowed_roots = roots
            if skills:
                skill_commands = build_skill_commands(skills)
                if self._chat_input:
                    merged = list(SLASH_COMMANDS) + skill_commands
                    self._chat_input.update_slash_commands(merged)
                else:
                    logger.debug(
                        "Skill discovery completed (%d skills) but chat input "
                        "not yet mounted; autocomplete deferred",
                        len(skills),
                    )
        except OSError:
            # Clear stale cache so /reload failures don't silently
            # leave old data in place.
            self._discovered_skills = []
            self._skill_allowed_roots = []
            logger.warning(
                "Filesystem error during skill discovery",
                exc_info=True,
            )
            self.notify(
                "Could not scan skill directories. "
                "Some /skill: commands may be unavailable.",
                severity="warning",
                timeout=6,
                markup=False,
            )
        except Exception:
            self._discovered_skills = []
            self._skill_allowed_roots = []
            logger.exception("Unexpected error during skill discovery")
            self.notify(
                "Skill discovery failed unexpectedly. "
                "/skill: commands may not work. Check logs for details.",
                severity="warning",
                timeout=8,
                markup=False,
            )

    def _discover_skills_and_roots(
        self,
    ) -> tuple[list[ExtendedSkillMetadata], list[Path]]:
        """기술을 발견하고 미리 해결된 격리 루트를 구축하세요.

        `list_skills` 호출 및 루트 해결 논리의 중복을 방지하기 위해 `_discover_skills`(시작/다시 로드) 및
        `_handle_skill_command`의 캐시 누락 폴백에서 공유됩니다.

Returns:
            `(skill metadata list, pre-resolved containment roots)`의 튜플입니다.

        """
        from deepagents_cli.config import settings
        from deepagents_cli.skills.load import list_skills

        assistant_id = self._assistant_id or "agent"
        skills = list_skills(
            built_in_skills_dir=settings.get_built_in_skills_dir(),
            user_skills_dir=settings.get_user_skills_dir(assistant_id),
            project_skills_dir=settings.get_project_skills_dir(),
            user_agent_skills_dir=settings.get_user_agent_skills_dir(),
            project_agent_skills_dir=settings.get_project_agent_skills_dir(),
            user_claude_skills_dir=settings.get_user_claude_skills_dir(),
            project_claude_skills_dir=settings.get_project_claude_skills_dir(),
        )
        # Pre-resolve containment roots once so _handle_skill_command
        # doesn't repeat resolve() on every invocation.
        roots = [
            d.resolve()
            for d in (
                settings.get_built_in_skills_dir(),
                settings.get_user_skills_dir(assistant_id),
                settings.get_project_skills_dir(),
                settings.get_user_agent_skills_dir(),
                settings.get_project_agent_skills_dir(),
                settings.get_user_claude_skills_dir(),
                settings.get_project_claude_skills_dir(),
            )
            if d is not None
        ]
        # Extra dirs are containment-only (not discovery); they allow
        # symlinks in standard dirs to point outside those dirs.
        roots.extend(d.resolve() for d in settings.get_extra_skills_dirs())
        return skills, roots

    async def _resolve_resume_thread(self) -> None:
        """`-r` 재개 의도를 구체적인 스레드 ID로 해결합니다.

        `self._resume_thread_intent`을(를) 소비하고 이를 구체적인 스레드 ID로 확인합니다.
        `self._lc_thread_id` 및 선택적으로 `self._assistant_id` / `self._server_kwargs`을
        변형합니다. DB 오류가 발생하면 새로운 스레드로 대체됩니다.

        """
        from deepagents_cli.sessions import (
            find_similar_threads,
            generate_thread_id,
            get_most_recent,
            get_thread_agent,
            thread_exists,
        )

        resume = self._resume_thread_intent
        self._resume_thread_intent = None  # consumed

        if not resume:
            return

        # Matches _DEFAULT_AGENT_NAME in main.py. Do NOT import it — main.py is
        # the CLI entry point and pulls in argparse, rich, etc. at module level.
        # Even a deferred import drags in the full dep tree for a single
        # string constant.
        default_agent = "agent"

        try:
            if resume == "__MOST_RECENT__":
                agent_filter = (
                    self._assistant_id if self._assistant_id != default_agent else None
                )
                thread_id = await get_most_recent(agent_filter)
                if thread_id:
                    agent_name = await get_thread_agent(thread_id)
                    if agent_name:
                        self._assistant_id = agent_name
                        if self._server_kwargs:
                            self._server_kwargs["assistant_id"] = agent_name
                    self._lc_thread_id = thread_id
                else:
                    self._lc_thread_id = generate_thread_id()
                    if agent_filter:
                        msg = f"No previous threads for '{agent_filter}', starting new."
                    else:
                        msg = "No previous threads, starting new."
                    self.notify(msg, severity="warning", markup=False)
            elif await thread_exists(resume):
                self._lc_thread_id = resume
                if self._assistant_id == default_agent:
                    agent_name = await get_thread_agent(resume)
                    if agent_name:
                        self._assistant_id = agent_name
                        if self._server_kwargs:
                            self._server_kwargs["assistant_id"] = agent_name
            else:
                # Thread not found — notify + fall back to new thread
                self._lc_thread_id = generate_thread_id()
                similar = await find_similar_threads(resume)
                hint = f"Thread '{resume}' not found."
                if similar:
                    hint += f" Did you mean: {', '.join(str(t) for t in similar)}?"
                self.notify(hint, severity="warning", timeout=6, markup=False)
        except Exception:
            logger.exception("Failed to resolve resume thread %r", resume)
            self._lc_thread_id = generate_thread_id()
            self.notify(
                "Could not look up thread history. Starting new session.",
                severity="warning",
            )

        # Update session state if ready (may still be initializing in a
        # concurrent worker)
        if self._session_state:
            self._session_state.thread_id = self._lc_thread_id

    async def _start_server_background(self) -> None:
        """백그라운드 작업자: 재개 스레드 의도를 해결하고 서버 시작 + MCP 사전 로드.

        또한 `model_kwargs`이 제공된 경우 지연된 모델 생성을 실행하므로 langchain import + init가 첫 번째 페인트를
        차단하지 않습니다.

        """
        # Phase 1: Resolve resume thread (if any) before server startup
        if self._resume_thread_intent:
            await self._resolve_resume_thread()

        # Run deferred model creation. settings.model_name / model_provider
        # are already set eagerly for the status bar display; this call
        # does the heavy langchain import + SDK init and may refine them
        # (e.g., context_limit from the model profile).
        if self._model_kwargs is not None:
            from deepagents_cli.config import create_model
            from deepagents_cli.model_config import ModelConfigError, save_recent_model

            try:
                result = create_model(**self._model_kwargs)
            except ModelConfigError as exc:
                self.post_message(self.ServerStartFailed(error=exc))
                return
            result.apply_to_settings()
            save_recent_model(f"{result.provider}:{result.model_name}")
            self._model_kwargs = None  # consumed

        from deepagents_cli.server_manager import start_server_and_get_agent

        coros: list[Any] = [start_server_and_get_agent(**self._server_kwargs)]  # type: ignore[arg-type]

        if self._mcp_preload_kwargs is not None:
            from deepagents_cli.main import _preload_session_mcp_server_info

            coros.append(_preload_session_mcp_server_info(**self._mcp_preload_kwargs))

        try:
            results = await asyncio.gather(*coros, return_exceptions=True)
        except Exception as exc:  # noqa: BLE001  # defensive catch around gather
            self.post_message(self.ServerStartFailed(error=exc))
            return

        server_result = results[0]
        if isinstance(server_result, BaseException):
            self.post_message(
                self.ServerStartFailed(
                    error=server_result
                    if isinstance(server_result, Exception)
                    else RuntimeError(str(server_result)),
                )
            )
            return

        agent, server_proc, _ = server_result

        # Assign immediately so the finally block in run_textual_app can
        # clean up the server even if the ServerReady message is never
        # processed (e.g. user quits during startup).
        self._server_proc = server_proc

        mcp_info = None
        if len(results) > 1 and not isinstance(results[1], BaseException):
            mcp_info = results[1]
        elif len(results) > 1 and isinstance(results[1], BaseException):
            logger.warning(
                "MCP metadata preload failed: %s",
                results[1],
                exc_info=results[1],
            )

        self.post_message(
            self.ServerReady(
                agent=agent,
                server_proc=server_proc,
                mcp_server_info=mcp_info,
            )
        )

    def on_deep_agents_app_server_ready(self, event: ServerReady) -> None:
        """성공적인 백그라운드 서버 시작을 처리합니다."""
        self._connecting = False
        self._agent = event.agent
        self._server_proc = event.server_proc
        self._mcp_server_info = event.mcp_server_info
        self._mcp_tool_count = sum(len(s.tools) for s in (event.mcp_server_info or []))

        # Update welcome banner to show ready state
        try:
            banner = self.query_one("#welcome-banner", WelcomeBanner)
            banner.set_connected(self._mcp_tool_count)
        except NoMatches:
            logger.warning("Welcome banner not found during server ready transition")

        # Handle deferred initial prompt or thread history
        if self._initial_prompt and self._initial_prompt.strip():
            prompt = self._initial_prompt
            self.call_after_refresh(
                lambda: asyncio.create_task(self._handle_user_message(prompt))
            )
        elif self._lc_thread_id and self._agent:
            self.call_after_refresh(
                lambda: asyncio.create_task(self._load_thread_history())
            )

        # Drain deferred actions (e.g. model/thread switch queued during connection)
        # if the agent is not actively running. Wrapped in a helper so that
        # exceptions are logged rather than becoming unhandled task errors.
        if self._deferred_actions and not self._agent_running:

            async def _safe_drain() -> None:
                try:
                    await self._maybe_drain_deferred()
                except Exception:
                    logger.exception("Unhandled error while draining deferred actions")
                    with suppress(Exception):
                        await self._mount_message(
                            ErrorMessage(
                                "A deferred action failed during startup. "
                                "You may need to retry the operation."
                            )
                        )

            self.call_after_refresh(lambda: asyncio.create_task(_safe_drain()))

        # Drain any messages the user typed while the server was starting.
        # (If an initial prompt exists, its cleanup path will drain the queue.)
        if self._pending_messages and not (
            self._initial_prompt and self._initial_prompt.strip()
        ):
            self.call_after_refresh(
                lambda: asyncio.create_task(self._process_next_from_queue())
            )

    def on_deep_agents_app_server_start_failed(self, event: ServerStartFailed) -> None:
        """백그라운드 서버 시작 실패를 처리합니다."""
        self._connecting = False
        logger.error("Server startup failed: %s", event.error, exc_info=event.error)
        # Update banner to show persistent failure state
        try:
            banner = self.query_one("#welcome-banner", WelcomeBanner)
            banner.set_failed(str(event.error))
        except NoMatches:
            logger.warning("Welcome banner not found during server failure transition")

        # Discard any messages queued while the server was starting
        if self._pending_messages:
            self._pending_messages.clear()
            for w in self._queued_widgets:
                w.remove()
            self._queued_widgets.clear()
        self._deferred_actions.clear()

    @staticmethod
    def _prewarm_deferred_imports() -> None:
        """시작 경로에서 지연되는 백그라운드 로드 모듈입니다.

        `sys.modules`을 채워 첫 번째 사용자 트리거 인라인 가져오기가 콜드 모듈 로드 대신 저렴한 dict 조회가 되도록 합니다.

        """
        # Internal modules moved from top-level to local imports — a failure
        # here indicates a packaging or code bug, not a missing optional dep, so
        # we let the exception propagate (the worker catches it and logs
        # at WARNING). textual_adapter and update_check are included so
        # _post_paint_init's inline imports are dict lookups.
        from deepagents_cli.clipboard import (
            copy_selection_to_clipboard,  # noqa: F401
        )
        from deepagents_cli.command_registry import ALWAYS_IMMEDIATE  # noqa: F401
        from deepagents_cli.config import settings  # noqa: F401
        from deepagents_cli.hooks import dispatch_hook  # noqa: F401
        from deepagents_cli.model_config import ModelSpec  # noqa: F401
        from deepagents_cli.textual_adapter import TextualUIAdapter  # noqa: F401
        from deepagents_cli.update_check import is_update_check_enabled  # noqa: F401

        try:
            # Heavy third-party deps deferred from textual_adapter /
            # tool_display — hit on first message send and first tool
            # approval. Best-effort: missing optional deps should not block the
            # TUI from rendering.
            from deepagents.backends import DEFAULT_EXECUTE_TIMEOUT  # noqa: F401
            from langchain.agents.middleware.human_in_the_loop import (  # noqa: F401
                ApproveDecision,
            )
            from langchain_core.messages import AIMessage  # noqa: F401
            from langgraph.types import Command  # noqa: F401
        except Exception:
            logger.warning("Could not prewarm third-party imports", exc_info=True)

        # Markdown rendering stack — ~170 ms cold (textual._markdown pulls in
        # markdown_it, pygments, linkify_it — 438 modules).  Hit on first
        # SkillMessage compose() and first code-fence highlight.  Warming
        # here makes the first expand/Ctrl+O instant.
        import markdown_it  # noqa: F401
        from pygments.lexers import get_lexer_by_name as _get_lexer
        from textual.widgets import Markdown  # noqa: F401

        # Instantiate the Python lexer to populate Pygments' internal
        # lexer cache (~12 ms cold).  Python is the most common fence
        # language in skill bodies.
        _get_lexer("python")

        # Widgets deferred from app.py module level — a failure here indicates
        # a packaging or code bug (same as the block above), so we let
        # exceptions propagate.
        from deepagents_cli.widgets.approval import ApprovalMenu  # noqa: F401
        from deepagents_cli.widgets.ask_user import AskUserMenu  # noqa: F401
        from deepagents_cli.widgets.model_selector import (
            ModelSelectorScreen,  # noqa: F401
        )
        from deepagents_cli.widgets.thread_selector import (  # noqa: F401
            DeleteThreadConfirmScreen,
            ThreadSelectorScreen,
        )

    async def _prewarm_threads_cache(self) -> None:  # noqa: PLR6301  # Worker hook kept as instance method
        """앱 시작을 차단하지 않고 미리 준비한 스레드 선택기 캐시입니다."""
        from deepagents_cli.sessions import (
            get_thread_limit,
            prewarm_thread_message_counts,
        )

        await prewarm_thread_message_counts(limit=get_thread_limit())

    async def _prewarm_model_caches(self) -> None:
        """시작을 차단하지 않고 사전 준비 모델 검색 및 프로필 캐시를 수행합니다."""
        try:
            from deepagents_cli.model_config import (
                get_available_models,
                get_model_profiles,
            )

            await asyncio.to_thread(get_available_models)
            await asyncio.to_thread(
                get_model_profiles, cli_override=self._profile_override
            )
        except Exception:
            logger.warning("Could not prewarm model caches", exc_info=True)

    async def _check_for_updates(self) -> None:
        """최신 버전이 있는지 PyPI를 확인하고 선택적으로 자동 업데이트하세요."""
        # Phase 1: version check (benign failure)
        try:
            from deepagents_cli.update_check import (
                is_auto_update_enabled,
                is_update_available,
                upgrade_command,
            )

            available, latest = await asyncio.to_thread(is_update_available)
            if not available:
                return

            self._update_available = (True, latest)
        except Exception:
            logger.debug("Background update check failed", exc_info=True)
            return

        # Phase 2: auto-update or notify (failures surfaced to user)
        try:
            from deepagents_cli._version import __version__ as cli_version

            if is_auto_update_enabled():
                from deepagents_cli.update_check import perform_upgrade

                self.notify(
                    f"Updating to v{latest}...",
                    severity="information",
                    timeout=5,
                )
                success, _output = await perform_upgrade()
                if success:
                    self.notify(
                        f"Updated to v{latest}. Restart to use the new version.",
                        severity="information",
                        timeout=10,
                    )
                else:
                    cmd = upgrade_command()
                    self.notify(
                        f"Auto-update failed. Run manually: {cmd}",
                        severity="warning",
                        timeout=15,
                        markup=False,
                    )
            else:
                cmd = upgrade_command()
                self.notify(
                    f"Update available: v{latest} (current: v{cli_version}). "
                    f"Run: {cmd}\n"
                    f"Enable auto-updates: /auto-update",
                    severity="information",
                    timeout=15,
                    markup=False,
                )
        except Exception:
            logger.warning("Auto-update failed unexpectedly", exc_info=True)
            self.notify(
                "Update failed unexpectedly.",
                severity="warning",
                timeout=10,
            )

    async def _show_whats_new(self) -> None:
        """업그레이드 후 처음 실행 시 '새로운 기능' 배너를 표시합니다."""
        try:
            from deepagents_cli.update_check import should_show_whats_new

            if not await asyncio.to_thread(should_show_whats_new):
                return
        except Exception:
            logger.debug("What's new check failed", exc_info=True)
            return

        try:
            from deepagents_cli._version import __version__ as cli_version

            await self._mount_message(
                AppMessage(
                    f"Updated to v{cli_version}\nSee what's new: {CHANGELOG_URL}"
                )
            )
        except Exception:
            logger.debug("What's new banner display failed", exc_info=True)
            return

        try:
            from deepagents_cli._version import __version__ as cli_version
            from deepagents_cli.update_check import mark_version_seen

            await asyncio.to_thread(mark_version_seen, cli_version)
        except Exception:
            logger.warning("Failed to persist seen-version marker", exc_info=True)

    async def _handle_update_command(self) -> None:
        """`/update` 슬래시 명령을 처리합니다. 업데이트를 확인하고 설치합니다."""

        await self._mount_message(UserMessage("/update"))
        try:
            from deepagents_cli.update_check import (
                is_update_available,
                perform_upgrade,
                upgrade_command,
            )

            await self._mount_message(AppMessage("Checking for updates..."))
            available, latest = await asyncio.to_thread(
                is_update_available, bypass_cache=True
            )
            if not available:
                await self._mount_message(AppMessage("Already on the latest version."))
                return

            from deepagents_cli._version import __version__ as cli_version

            await self._mount_message(
                AppMessage(
                    f"Update available: v{latest} (current: v{cli_version}). "
                    "Upgrading..."
                )
            )
            success, output = await perform_upgrade()
            if success:
                self._update_available = (False, None)
                await self._mount_message(
                    AppMessage(f"Updated to v{latest}. Restart to use the new version.")
                )
            else:
                cmd = upgrade_command()
                detail = f": {output[:200]}" if output else ""
                await self._mount_message(
                    AppMessage(f"Auto-update failed{detail}\nRun manually: {cmd}")
                )
        except Exception as exc:
            logger.warning("/update command failed", exc_info=True)
            await self._mount_message(
                ErrorMessage(f"Update failed: {type(exc).__name__}: {exc}")
            )

    async def _handle_auto_update_toggle(self) -> None:
        """`/auto-update` 슬래시 명령을 처리합니다. 즉시 토글을 지속합니다."""

        try:
            from deepagents_cli.config import _is_editable_install
            from deepagents_cli.update_check import (
                is_auto_update_enabled,
                set_auto_update,
            )

            if await asyncio.to_thread(_is_editable_install):
                self.notify(
                    "Auto-updates are not available for editable installs.",
                    severity="warning",
                    timeout=5,
                )
                return

            currently_enabled = await asyncio.to_thread(is_auto_update_enabled)
            new_state = not currently_enabled
            await asyncio.to_thread(set_auto_update, new_state)
            label = "enabled" if new_state else "disabled"
            self.notify(
                f"Auto-updates {label}.",
                severity="information",
                timeout=5,
                markup=False,
            )
        except Exception as exc:
            logger.warning("/auto-update command failed", exc_info=True)
            self.notify(
                f"Auto-update toggle failed: {type(exc).__name__}: {exc}",
                severity="warning",
                timeout=5,
                markup=False,
            )

    def on_scroll_up(self, _event: ScrollUp) -> None:
        """위로 스크롤하여 오래된 메시지를 수화해야 하는지 확인하세요."""
        self._check_hydration_needed()

    def _update_status(self, message: str) -> None:
        """메시지로 상태 표시줄을 업데이트합니다."""
        if self._status_bar:
            self._status_bar.set_status_message(message)

    def _update_tokens(self, count: int, *, approximate: bool = False) -> None:
        """상태 표시줄에서 토큰 수를 업데이트합니다.

        낮은 수준의 도우미 — UI에만 닿습니다.  로컬 캐시도 업데이트해야 하는 호출자는 대신 `_on_tokens_update`을 사용해야 합니다.

Args:
            count: 총 컨텍스트 토큰 수입니다.
            approximate: 오래된/중단된 카운트를 알리려면 "+"를 추가하세요.

        """
        if self._status_bar:
            self._status_bar.set_tokens(count, approximate=approximate)

    def _on_tokens_update(self, count: int, *, approximate: bool = False) -> None:
        """로컬 캐시 *및* 상태 표시줄을 업데이트합니다.

        이는 어댑터의 `_on_tokens_update`에 연결된 콜백입니다.

Args:
            count: 캐시하고 표시할 총 컨텍스트 토큰 수입니다.
            approximate: 오래된/중단된 카운트를 알리려면 "+"를 추가하세요.

        """
        self._context_tokens = count
        self._tokens_approximate = approximate
        self._update_tokens(count, approximate=approximate)

    def _show_tokens(self, *, approximate: bool = False) -> None:
        """상태 표시줄을 캐시된 토큰 값으로 복원합니다.

Args:
            approximate: 오래된/중단된 카운트를 알리려면 "+"를 추가하세요.

                이 플래그는 `_on_tokens_update`이 모델로부터 새로운 카운트를 받을 때까지 고정되어 있습니다.

        """
        self._tokens_approximate = self._tokens_approximate or approximate
        self._update_tokens(
            self._context_tokens,
            approximate=self._tokens_approximate,
        )

    def _hide_tokens(self) -> None:
        """스트리밍 중에 토큰 표시를 숨깁니다."""
        if self._status_bar:
            self._status_bar.hide_tokens()

    def _check_hydration_needed(self) -> None:
        """매장에서 보낸 메시지를 수화해야 하는지 확인하세요.

        사용자가 표시되는 메시지 상단 근처에서 위로 스크롤할 때 호출됩니다.

        """
        if not self._message_store.has_messages_above:
            return

        try:
            chat = self.query_one("#chat", VerticalScroll)
        except NoMatches:
            logger.debug("Skipping hydration check: #chat container not found")
            return

        scroll_y = chat.scroll_y
        viewport_height = chat.size.height

        if self._message_store.should_hydrate_above(scroll_y, viewport_height):
            self.call_later(self._hydrate_messages_above)

    async def _hydrate_messages_above(self) -> None:
        """사용자가 상단 근처로 스크롤하면 오래된 메시지를 수화합니다.

        그러면 보관된 메시지에 대한 위젯이 다시 생성되어 메시지 컨테이너 상단에 삽입됩니다.

        """
        if not self._message_store.has_messages_above:
            return

        try:
            chat = self.query_one("#chat", VerticalScroll)
        except NoMatches:
            logger.debug("Skipping hydration: #chat not found")
            return

        try:
            messages_container = self.query_one("#messages", Container)
        except NoMatches:
            logger.debug("Skipping hydration: #messages not found")
            return

        to_hydrate = self._message_store.get_messages_to_hydrate()
        if not to_hydrate:
            return

        old_scroll_y = chat.scroll_y
        first_child = (
            messages_container.children[0] if messages_container.children else None
        )

        # Build widgets in chronological order, then mount in reverse so
        # each is inserted before the previous first_child, resulting in
        # correct chronological order in the DOM.
        hydrated_count = 0
        hydrated_widgets: list[tuple] = []  # (widget, msg_data)
        for msg_data in to_hydrate:
            try:
                widget = msg_data.to_widget()
                hydrated_widgets.append((widget, msg_data))
            except Exception:
                logger.warning(
                    "Failed to create widget for message %s",
                    msg_data.id,
                    exc_info=True,
                )

        for widget, msg_data in reversed(hydrated_widgets):
            try:
                if first_child:
                    await messages_container.mount(widget, before=first_child)
                else:
                    await messages_container.mount(widget)
                first_child = widget
                hydrated_count += 1
                # Render Markdown content for hydrated assistant messages
                if isinstance(widget, AssistantMessage) and msg_data.content:
                    await widget.set_content(msg_data.content)
            except Exception:
                logger.warning(
                    "Failed to mount hydrated widget %s",
                    widget.id,
                    exc_info=True,
                )

        # Only update store for the number we actually mounted
        if hydrated_count > 0:
            self._message_store.mark_hydrated(hydrated_count)

        # Adjust scroll position to maintain the user's view.
        # Widget heights aren't known until after layout, so we use a
        # heuristic. A more accurate approach would measure actual heights
        # via call_after_refresh.
        estimated_height_per_message = 5  # terminal rows, rough estimate
        added_height = hydrated_count * estimated_height_per_message
        chat.scroll_y = old_scroll_y + added_height

    async def _mount_before_queued(self, container: Container, widget: Widget) -> None:
        """대기 중인 위젯 이전에 메시지 컨테이너에 위젯을 마운트합니다.

        대기 중인 메시지 위젯은 현재 에이전트 응답 아래에 시각적으로 고정되어 있도록 컨테이너 하단에 있어야 합니다. 이 도우미는 첫 번째 대기열에
        추가된 위젯 바로 앞에 `widget`을 삽입하거나 대기열이 비어 있을 때 끝에 추가합니다.

Args:
            container: 마운트할 `#messages` 컨테이너입니다.
            widget: 마운트할 위젯입니다.

        """
        if not container.is_attached:
            return
        first_queued = self._queued_widgets[0] if self._queued_widgets else None
        if first_queued is not None and first_queued.parent is container:
            try:
                await container.mount(widget, before=first_queued)
            except Exception:
                logger.warning(
                    "Stale queued-widget reference; appending at end",
                    exc_info=True,
                )
            else:
                return
        await container.mount(widget)

    def _is_spinner_at_correct_position(self, container: Container) -> bool:
        """로딩 스피너가 이미 올바르게 배치되었는지 확인하세요.

        스피너는 첫 번째 대기열에 추가된 위젯 바로 앞에 있거나 대기열이 비어 있을 때 컨테이너 맨 끝에 있어야 합니다.

Args:
            container: `#messages` 컨테이너.

Returns:
            `True` 스피너가 이미 올바른 위치에 있는 경우.

        """
        children = list(container.children)
        if not children or self._loading_widget not in children:
            return False

        if self._queued_widgets:
            first_queued = self._queued_widgets[0]
            if first_queued not in children:
                return False
            return children.index(self._loading_widget) == (
                children.index(first_queued) - 1
            )

        return children[-1] == self._loading_widget

    async def _set_spinner(self, status: SpinnerStatus) -> None:
        """로딩 스피너를 표시, 업데이트 또는 숨깁니다.

Args:
            status: 표시하려면 스피너 상태이고, 숨기려면 `None`입니다.

        """
        if status is None:
            # Hide
            if self._loading_widget:
                await self._loading_widget.remove()
                self._loading_widget = None
            return

        messages = self.query_one("#messages", Container)

        if self._loading_widget is None:
            # Create new
            self._loading_widget = LoadingWidget(status)
            await self._mount_before_queued(messages, self._loading_widget)
        else:
            # Update existing
            self._loading_widget.set_status(status)
            # Reposition if not already at the correct location
            if not self._is_spinner_at_correct_position(messages):
                await self._loading_widget.remove()
                await self._mount_before_queued(messages, self._loading_widget)
        # NOTE: Don't call anchor() here - it would re-anchor and drag user back
        # to bottom if they've scrolled away during streaming

    async def _request_approval(
        self,
        action_requests: Any,  # noqa: ANN401  # ActionRequest uses dynamic typing
        assistant_id: str | None,
    ) -> asyncio.Future:
        """메시지 영역에서 인라인으로 사용자 승인을 요청하세요.

        메시지 영역에 ApprovalMenu를 탑재합니다(채팅과 함께 인라인). ChatInput은 계속 표시됩니다. 사용자는 계속 볼 수 있습니다.

        다른 승인이 이미 보류 중인 경우 이 승인을 대기열에 추가하세요.

        구성된 허용 목록에 있는 셸 명령을 자동 승인합니다.

Args:
            action_requests: 승인할 작업 요청 목록
            assistant_id: 표시 목적의 어시스턴트 ID

Returns:
            사용자의 결정으로 결정되는 Future입니다.

        """
        from deepagents_cli.config import (
            SHELL_TOOL_NAMES,
            is_shell_command_allowed,
            settings,
        )

        loop = asyncio.get_running_loop()
        result_future: asyncio.Future = loop.create_future()

        # Check if ALL actions in the batch are auto-approvable shell commands
        if settings.shell_allow_list and action_requests:
            all_auto_approved = True
            approved_commands = []

            for req in action_requests:
                if req.get("name") in SHELL_TOOL_NAMES:
                    command = req.get("args", {}).get("command", "")
                    if is_shell_command_allowed(command, settings.shell_allow_list):
                        approved_commands.append(command)
                    else:
                        all_auto_approved = False
                        break
                else:
                    # Non-shell commands need normal approval
                    all_auto_approved = False
                    break

            if all_auto_approved and approved_commands:
                # Auto-approve all commands in the batch
                result_future.set_result({"type": "approve"})

                # Mount system messages showing the auto-approvals
                try:
                    messages = self.query_one("#messages", Container)
                    for command in approved_commands:
                        auto_msg = AppMessage(
                            f"✓ Auto-approved shell command (allow-list): {command}"
                        )
                        await self._mount_before_queued(messages, auto_msg)
                    with suppress(NoMatches, ScreenStackError):
                        self.query_one("#chat", VerticalScroll).anchor()
                except Exception:  # noqa: S110, BLE001  # Resilient auto-message display
                    pass  # Don't fail if we can't show the message

                return result_future

        # If there's already a pending approval, wait for it to complete first
        if self._pending_approval_widget is not None:
            while self._pending_approval_widget is not None:  # noqa: ASYNC110  # Simple polling is sufficient here
                await asyncio.sleep(0.1)

        # Create menu with unique ID to avoid conflicts
        from deepagents_cli.widgets.approval import ApprovalMenu

        unique_id = f"approval-menu-{uuid.uuid4().hex[:8]}"
        menu = ApprovalMenu(action_requests, assistant_id, id=unique_id)
        menu.set_future(result_future)

        self._pending_approval_widget = menu

        if self._is_user_typing():
            # Show a placeholder until the user stops typing, then swap in the
            # real ApprovalMenu.  This prevents accidental key presses (e.g.
            # 'y', 'n') from triggering approval decisions mid-sentence.
            placeholder = Static(
                "Waiting for typing to finish...",
                classes="approval-placeholder",
            )
            self._approval_placeholder = placeholder
            try:
                messages = self.query_one("#messages", Container)
                await self._mount_before_queued(messages, placeholder)
                self.call_after_refresh(placeholder.scroll_visible)
            except Exception:
                logger.exception("Failed to mount approval placeholder")
                # Placeholder failed — fall back to showing the menu directly
                # so the future is always resolvable.
                self._approval_placeholder = None
                await self._mount_approval_widget(menu, result_future)
                return result_future

            self.run_worker(
                self._deferred_show_approval(placeholder, menu, result_future),
                exclusive=False,
            )
        else:
            await self._mount_approval_widget(menu, result_future)

        return result_future

    async def _mount_approval_widget(
        self,
        menu: ApprovalMenu,
        result_future: asyncio.Future[dict[str, str]],
    ) -> None:
        """메시지 영역에 승인 메뉴 위젯을 인라인으로 탑재합니다.

        마운트에 실패하면 `_pending_approval_widget`을 지우고 `result_future`을 통해 예외를 전파합니다.

Args:
            menu: 마운트할 `ApprovalMenu` 인스턴스입니다.
            result_future: 호출자에 대해 해결/거부할 미래입니다.

        """
        try:
            messages = self.query_one("#messages", Container)
            await self._mount_before_queued(messages, menu)
            self.call_after_refresh(menu.scroll_visible)
            self.call_after_refresh(menu.focus)
        except Exception as e:
            logger.exception(
                "Failed to mount approval menu (id=%s) in messages container",
                menu.id,
            )
            self._pending_approval_widget = None
            if not result_future.done():
                result_future.set_exception(e)

    async def _deferred_show_approval(
        self,
        placeholder: Static,
        menu: ApprovalMenu,
        result_future: asyncio.Future[dict[str, str]],
    ) -> None:
        """사용자가 유휴 상태가 될 때까지 기다린 다음 자리 표시자를 실제 메뉴로 바꿉니다.

        자리 표시자가 이미 분리된 경우 조기 종료됩니다(예: 대기 중에 승인이 취소됨).  이 경우 future가 취소되어 호출자가 계속 대기 상태로
        남지 않습니다.

Args:
            placeholder: 현재 마운트된 임시 자리 표시자 위젯입니다.
            menu: 사용자가 입력을 중지하면 표시되는 `ApprovalMenu`입니다.
            result_future: 이 승인 흐름을 뒷받침하는 미래.

        """
        deadline = _monotonic() + _DEFERRED_APPROVAL_TIMEOUT_SECONDS
        while self._is_user_typing():  # Simple polling
            if _monotonic() > deadline:
                logger.warning(
                    "Timed out waiting for user to stop typing; showing approval now"
                )
                break
            await asyncio.sleep(0.2)

        # Guard: if the placeholder was already removed (e.g. agent cancelled
        # the approval while we were waiting), clean up and cancel the future.
        if not placeholder.is_attached:
            logger.warning(
                "Approval placeholder detached before menu shown (id=%s)",
                menu.id,
            )
            self._approval_placeholder = None
            self._pending_approval_widget = None
            if not result_future.done():
                result_future.cancel()
            return

        self._approval_placeholder = None
        try:
            await placeholder.remove()
        except Exception:
            logger.warning(
                "Failed to remove approval placeholder during swap",
                exc_info=True,
            )
        await self._mount_approval_widget(menu, result_future)

    def _on_auto_approve_enabled(self) -> None:
        """HITL 승인 메뉴를 통해 활성화되는 자동 승인을 처리합니다.

        사용자가 승인 대화상자에서 "모두 자동 승인"을 선택하면 호출됩니다. 앱 플래그, 상태 표시줄 표시기 및 세션 상태 전반에 걸쳐 자동 승인
        상태를 동기화하므로 후속 도구 호출에서는 승인 프롬프트를 건너뛸 수 있습니다.

        """
        self._auto_approve = True
        if self._status_bar:
            self._status_bar.set_auto_approve(enabled=True)
        if self._session_state:
            self._session_state.auto_approve = True

    async def _remove_ask_user_widget(  # noqa: PLR6301  # Shared helper used by ask_user event handlers
        self,
        widget: AskUserMenu,
        *,
        context: str,
    ) -> None:
        """정리 경주를 표시하지 않고 Ask_user 위젯을 제거합니다.

Args:
            widget: 사용자에게 제거할 위젯 인스턴스를 요청합니다.
            context: 진단을 위한 짧은 컨텍스트 문자열입니다.

        """
        try:
            await widget.remove()
        except Exception:
            logger.debug(
                "Failed to remove ask-user widget during %s",
                context,
                exc_info=True,
            )

    async def _request_ask_user(
        self,
        questions: list[Question],
    ) -> asyncio.Future[AskUserWidgetResult]:
        """Ask_user 위젯을 표시하고 사용자 응답과 함께 Future를 반환합니다.

Args:
            questions: 각각 `question`, `type`, 선택적 `choices` 및 `required` 키가 포함된 질문 사전
                       목록입니다.

Returns:
            `'type'`(`'answered'` 또는
                `'cancelled'`) 및 응답 시 `'answers'` 목록.

        """
        loop = asyncio.get_running_loop()
        result_future: asyncio.Future[AskUserWidgetResult] = loop.create_future()

        if self._pending_ask_user_widget is not None:
            deadline = _monotonic() + 30
            while self._pending_ask_user_widget is not None:
                if _monotonic() > deadline:
                    logger.error(
                        "Timed out waiting for previous ask-user widget to "
                        "clear. Forcefully cleaning up."
                    )
                    old_widget = self._pending_ask_user_widget
                    if old_widget is not None:
                        old_widget.action_cancel()
                        self._pending_ask_user_widget = None
                        await self._remove_ask_user_widget(
                            old_widget,
                            context="ask-user timeout cleanup",
                        )
                    break
                await asyncio.sleep(0.1)

        from deepagents_cli.widgets.ask_user import AskUserMenu

        unique_id = f"ask-user-menu-{uuid.uuid4().hex[:8]}"
        menu = AskUserMenu(questions, id=unique_id)
        menu.set_future(result_future)

        self._pending_ask_user_widget = menu

        try:
            messages = self.query_one("#messages", Container)
            await self._mount_before_queued(messages, menu)
            self.call_after_refresh(menu.scroll_visible)
            self.call_after_refresh(menu.focus_active)
        except Exception as e:
            logger.exception(
                "Failed to mount ask-user menu (id=%s)",
                unique_id,
            )
            self._pending_ask_user_widget = None
            if not result_future.done():
                result_future.set_exception(e)

        return result_future

    async def on_ask_user_menu_answered(
        self,
        event: Any,  # noqa: ARG002, ANN401
    ) -> None:
        """Ask_user 메뉴 답변 처리 - 위젯을 제거하고 입력에 다시 초점을 맞춥니다."""
        if self._pending_ask_user_widget:
            widget = self._pending_ask_user_widget
            self._pending_ask_user_widget = None
            await self._remove_ask_user_widget(widget, context="ask-user answered")

        if self._chat_input:
            self.call_after_refresh(self._chat_input.focus_input)

    async def on_ask_user_menu_cancelled(
        self,
        event: Any,  # noqa: ARG002, ANN401
    ) -> None:
        """Ask_user 메뉴 취소 처리 - 위젯을 제거하고 입력에 다시 초점을 맞춥니다."""
        if self._pending_ask_user_widget:
            widget = self._pending_ask_user_widget
            self._pending_ask_user_widget = None
            await self._remove_ask_user_widget(widget, context="ask-user cancelled")

        if self._chat_input:
            self.call_after_refresh(self._chat_input.focus_input)

    async def _process_message(self, value: str, mode: InputMode) -> None:
        """모드에 따라 메시지를 적절한 핸들러로 라우팅합니다.

Args:
            value: 처리할 메시지 텍스트입니다.
            mode: 메시지 라우팅을 결정하는 입력 모드입니다.

        """
        if mode == "shell":
            await self._handle_shell_command(value.removeprefix("!"))
        elif mode == "command":
            await self._handle_command(value)
        elif mode == "normal":
            await self._handle_user_message(value)
        else:
            logger.warning("Unrecognized input mode %r, treating as normal", mode)
            await self._handle_user_message(value)

    def _can_bypass_queue(self, value: str) -> bool:
        """슬래시 명령이 메시지 대기열을 건너뛸 수 있는지 확인하세요.

Args:
            value: 낮춰지고 제거된 명령 문자열(예: `/model`)입니다.

Returns:
            `True` 명령이 사용 중 상태 대기열을 우회해야 하는 경우.

        """
        from deepagents_cli.command_registry import (
            BYPASS_WHEN_CONNECTING,
            IMMEDIATE_UI,
            SIDE_EFFECT_FREE,
        )

        cmd = value.split(maxsplit=1)[0] if value else ""
        if cmd in BYPASS_WHEN_CONNECTING:
            return self._connecting and not (self._agent_running or self._shell_running)
        if cmd in IMMEDIATE_UI:
            # Only bare form (no args) bypasses — /model opens selector,
            # /model <name> does a direct switch that shouldn't race with agent.
            return value == cmd
        return cmd in SIDE_EFFECT_FREE

    async def on_chat_input_submitted(self, event: ChatInput.Submitted) -> None:
        """ChatInput 위젯에서 제출된 입력을 처리합니다."""
        value = event.value
        mode: InputMode = event.mode  # type: ignore[assignment]  # Textual event mode is str at type level but InputMode at runtime

        # Reset quit pending state on any input
        self._quit_pending = False

        from deepagents_cli.hooks import dispatch_hook

        await dispatch_hook("user.prompt", {})

        # /quit and /q always execute immediately, even mid-thread-switch.
        from deepagents_cli.command_registry import ALWAYS_IMMEDIATE

        if mode == "command" and value.lower().strip() in ALWAYS_IMMEDIATE:
            self.exit()
            return

        # Prevent message handling while a thread switch is in-flight.
        if self._thread_switching:
            self.notify(
                "Thread switch in progress. Please wait.",
                severity="warning",
                timeout=3,
            )
            return

        # If agent/shell is running or server is still starting up, enqueue
        # instead of processing. Messages queued during connection are drained
        # once the server is ready (see on_deep_agents_app_server_ready).
        if self._agent_running or self._shell_running or self._connecting:
            if mode == "command" and self._can_bypass_queue(value.lower().strip()):
                await self._process_message(value, mode)
                return
            self._pending_messages.append(QueuedMessage(text=value, mode=mode))
            queued_widget = QueuedUserMessage(value)
            self._queued_widgets.append(queued_widget)
            await self._mount_message(queued_widget)
            return

        await self._process_message(value, mode)

    def on_chat_input_mode_changed(self, event: ChatInput.ModeChanged) -> None:
        """입력 모드가 변경되면 상태 표시줄을 업데이트합니다."""
        if self._status_bar:
            self._status_bar.set_mode(event.mode)

    def on_chat_input_typing(
        self,
        event: ChatInput.Typing,  # noqa: ARG002  # Textual event handler signature
    ) -> None:
        """입력 인식 승인 연기를 위해 가장 최근의 키 입력 시간을 기록합니다."""
        self._last_typed_at = _monotonic()

    def _is_user_typing(self) -> bool:
        """사용자가 최근에 입력했는지 여부를 반환합니다(유휴 임계값 내에서).

Returns:
            `True` 마지막으로 기록된 타이핑 이벤트가 마지막 기간 내에 발생한 경우
                `_TYPING_IDLE_THRESHOLD_SECONDS`초, 그렇지 않으면 `False`.

        """
        if self._last_typed_at is None:
            return False
        return (_monotonic() - self._last_typed_at) < _TYPING_IDLE_THRESHOLD_SECONDS

    async def on_approval_menu_decided(
        self,
        event: Any,  # noqa: ARG002, ANN401  # Textual event handler signature
    ) -> None:
        """승인 메뉴 결정 처리 - 메시지에서 제거하고 입력에 다시 집중합니다."""
        # Defensively remove any lingering placeholder (should already be gone
        # once the deferred worker swaps it, but guard against edge cases).
        if self._approval_placeholder is not None:
            if self._approval_placeholder.is_attached:
                try:
                    await self._approval_placeholder.remove()
                except Exception:
                    logger.warning(
                        "Failed to remove approval placeholder during cleanup",
                        exc_info=True,
                    )
            self._approval_placeholder = None

        # Remove ApprovalMenu using stored reference
        if self._pending_approval_widget:
            await self._pending_approval_widget.remove()
            self._pending_approval_widget = None

        # Refocus the chat input
        if self._chat_input:
            self.call_after_refresh(self._chat_input.focus_input)

    async def _handle_shell_command(self, command: str) -> None:
        """쉘 명령(! 접두사)을 처리합니다.

        사용자 메시지를 마운트하고 작업자를 생성하여 이벤트 루프가 주요 이벤트(Esc/Ctrl+C)에 대해 자유롭게 유지되도록 하는 씬 디스패처입니다.

Args:
            command: 실행할 쉘 명령입니다.

        """
        await self._mount_message(UserMessage(f"!{command}"))
        self._shell_running = True

        if self._chat_input:
            self._chat_input.set_cursor_active(active=False)

        self._shell_worker = self.run_worker(
            self._run_shell_task(command),
            exclusive=False,
        )

    async def _run_shell_task(self, command: str) -> None:
        """백그라운드 작업자에서 셸 명령을 실행합니다.

        이는 `_run_agent_task`을 미러링합니다. 작업자에서 실행하면 이벤트 루프를 자유롭게 유지하므로 Esc/Ctrl+C는 작업자 취소
        -> `CancelledError` 발생 -> 프로세스 종료를 수행할 수 있습니다.

Args:
            command: 실행할 쉘 명령입니다.

Raises:
            CancelledError: 사용자가 명령을 중단한 경우.

        """
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self._cwd,
                start_new_session=(sys.platform != "win32"),
            )
            self._shell_process = proc

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=60
                )
            except TimeoutError:
                await self._kill_shell_process()
                await self._mount_message(ErrorMessage("Command timed out (60s limit)"))
                return
            except asyncio.CancelledError:
                await self._kill_shell_process()
                raise

            output = (stdout_bytes or b"").decode(errors="replace").strip()
            stderr_text = (stderr_bytes or b"").decode(errors="replace").strip()
            if stderr_text:
                output += f"\n[stderr]\n{stderr_text}"

            if output:
                msg = AssistantMessage(f"```\n{output}\n```")
                await self._mount_message(msg)
                await msg.write_initial_content()
            else:
                await self._mount_message(AppMessage("Command completed (no output)"))

            if proc.returncode and proc.returncode != 0:
                await self._mount_message(ErrorMessage(f"Exit code: {proc.returncode}"))

            # Anchor to bottom so shell output stays visible
            with suppress(NoMatches, ScreenStackError):
                self.query_one("#chat", VerticalScroll).anchor()

        except OSError as e:
            logger.exception("Failed to execute shell command: %s", command)
            err_msg = f"Failed to run command: {e}"
            await self._mount_message(ErrorMessage(err_msg))
        finally:
            await self._cleanup_shell_task()

    async def _cleanup_shell_task(self) -> None:
        """셸 명령 작업이 완료되거나 취소된 후 정리합니다."""
        was_interrupted = self._shell_process is not None and (
            self._shell_worker is not None and self._shell_worker.is_cancelled
        )
        self._shell_process = None
        self._shell_running = False
        self._shell_worker = None
        if was_interrupted:
            await self._mount_message(AppMessage("Command interrupted"))
        if self._chat_input:
            self._chat_input.set_cursor_active(active=True)
        try:
            await self._maybe_drain_deferred()
        except Exception:
            logger.exception("Failed to drain deferred actions during shell cleanup")
            with suppress(Exception):
                await self._mount_message(
                    ErrorMessage(
                        "A deferred action failed after task completion. "
                        "You may need to retry the operation."
                    )
                )
        await self._process_next_from_queue()

    async def _kill_shell_process(self) -> None:
        """실행 중인 쉘 명령 프로세스를 종료합니다.

        POSIX에서는 SIGTERM을 전체 프로세스 그룹에 보냅니다(자식 종료). Windows에서는 루트 프로세스만 종료합니다. 프로세스가 이미
        종료된 경우 작동하지 않습니다. 완전한 종료를 위해 최대 5초를 기다린 후 SIGKILL로 에스컬레이션합니다.

        """
        proc = self._shell_process
        if proc is None or proc.returncode is not None:
            return

        try:
            if sys.platform != "win32":
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            else:
                proc.terminate()
        except ProcessLookupError:
            return
        except OSError:
            logger.warning(
                "Failed to terminate shell process (pid=%s)", proc.pid, exc_info=True
            )
            return

        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except TimeoutError:
            logger.warning(
                "Shell process (pid=%s) did not exit after SIGTERM; sending SIGKILL",
                proc.pid,
            )
            with suppress(ProcessLookupError, OSError):
                if sys.platform != "win32":
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                else:
                    proc.kill()
            with suppress(ProcessLookupError, OSError):
                await proc.wait()
        except (ProcessLookupError, OSError):
            pass

    async def _open_url_command(self, command: str, cmd: str) -> None:
        """브라우저에서 URL을 열고 클릭 가능한 링크를 표시합니다.

        사용 중인 상태와 관계없이 브라우저가 즉시 열립니다. 앱이 사용 중인 경우 대기열 표시기가 표시되고 현재 작업이 완료된 후 실제 채팅
        출력(사용자 에코 + 클릭 가능한 링크)이 이를 대체합니다.

Args:
            command: 원시 명령 텍스트(사용자 메시지로 표시됨)
            cmd: URL을 조회하는 데 사용되는 정규화된 슬래시 명령입니다.

        """
        url = _COMMAND_URLS[cmd]
        webbrowser.open(url)

        if self._agent_running or self._shell_running:
            queued_widget = QueuedUserMessage(command)
            self._queued_widgets.append(queued_widget)
            await self._mount_message(queued_widget)

            async def _mount_output() -> None:
                # Remove the ephemeral queued widget, then mount real output.
                if queued_widget in self._queued_widgets:
                    self._queued_widgets.remove(queued_widget)
                with suppress(Exception):
                    await queued_widget.remove()
                await self._mount_message(UserMessage(command))
                link = Content.styled(url, TStyle(dim=True, italic=True, link=url))
                await self._mount_message(AppMessage(link))

            # Append directly — no dedup; each URL command gets its own output.
            self._deferred_actions.append(
                DeferredAction(kind="chat_output", execute=_mount_output)
            )
            return

        await self._mount_message(UserMessage(command))
        link = Content.styled(url, TStyle(dim=True, italic=True, link=url))
        await self._mount_message(AppMessage(link))

    @staticmethod
    async def _build_thread_message(prefix: str, thread_id: str) -> str | Content:
        """가능하면 ID를 하이퍼링크하여 스레드 상태 메시지를 작성하십시오.

        짧은 시간 초과로 LangSmith 스레드 URL을 확인하려고 시도합니다. 추적이 구성되지 않았거나 확인에 실패하면 일반 텍스트로 대체됩니다.

Args:
            prefix: 스레드 ID 앞에 라벨을 붙입니다(예: `'Resumed thread'`).
            thread_id: 스레드 식별자입니다.

Returns:
            `Content`(클릭 가능한 스레드 ID 또는 일반 문자열)

        """
        from deepagents_cli.config import build_langsmith_thread_url

        try:
            url = await asyncio.wait_for(
                asyncio.to_thread(build_langsmith_thread_url, thread_id),
                timeout=2.0,
            )
        except (TimeoutError, Exception):  # noqa: BLE001  # Resilient non-interactive mode error handling
            url = None

        if url:
            return Content.assemble(
                f"{prefix}: ",
                (thread_id, TStyle(link=url)),
            )
        return f"{prefix}: {thread_id}"

    async def _handle_trace_command(self, command: str) -> None:
        """LangSmith에서 현재 스레드를 엽니다.

        사용 중인 상태와 관계없이 URL을 확인하고 즉시 브라우저를 엽니다. 앱이 사용 중이면 현재 작업이 완료될 때까지 채팅 출력(사용자 에코 +
        클릭 가능한 링크)이 연기됩니다. 오류 조건(세션 없음, URL 실패, 추적이 구성되지 않음)은 사용 중 상태와 관계없이 즉시 렌더링됩니다.

Args:
            command: 원시 명령 텍스트(사용자 메시지로 표시됨)

        """
        from deepagents_cli.config import build_langsmith_thread_url

        if not self._session_state:
            await self._mount_message(UserMessage(command))
            await self._mount_message(AppMessage("No active session."))
            return
        thread_id = self._session_state.thread_id
        try:
            url = await asyncio.to_thread(build_langsmith_thread_url, thread_id)
        except Exception:
            logger.exception("Failed to build LangSmith thread URL for %s", thread_id)
            await self._mount_message(UserMessage(command))
            await self._mount_message(
                AppMessage("Failed to resolve LangSmith thread URL.")
            )
            return
        if not url:
            await self._mount_message(UserMessage(command))
            await self._mount_message(
                AppMessage(
                    "LangSmith tracing is not configured. "
                    "Set LANGSMITH_API_KEY and LANGSMITH_TRACING=true to enable."
                )
            )
            return

        def _open_browser() -> None:
            try:
                webbrowser.open(url)
            except Exception:
                logger.debug("Could not open browser for URL: %s", url, exc_info=True)

        asyncio.get_running_loop().run_in_executor(None, _open_browser)

        # Defer chat output while a turn is in progress — rendering the user
        # echo + link immediately would splice it into the middle of the
        # streaming assistant response
        if self._agent_running or self._shell_running:
            queued_widget = QueuedUserMessage(command)
            self._queued_widgets.append(queued_widget)
            await self._mount_message(queued_widget)

            async def _mount_output() -> None:
                if queued_widget in self._queued_widgets:
                    self._queued_widgets.remove(queued_widget)
                with suppress(Exception):
                    await queued_widget.remove()
                await self._mount_message(UserMessage(command))
                link = Content.styled(url, TStyle(dim=True, italic=True, link=url))
                await self._mount_message(AppMessage(link))

            # Append directly — no dedup; each /trace invocation gets its own output.
            self._deferred_actions.append(
                DeferredAction(kind="chat_output", execute=_mount_output)
            )
            return

        await self._mount_message(UserMessage(command))
        link = Content.styled(url, TStyle(dim=True, italic=True, link=url))
        await self._mount_message(AppMessage(link))

    async def _handle_command(self, command: str) -> None:
        """슬래시 명령을 처리합니다.

Args:
            command: 슬래시 명령(/ 포함)

        """
        from deepagents_cli.config import newline_shortcut, settings

        cmd = command.lower().strip()

        if cmd in {"/quit", "/q"}:
            self.exit()
        elif cmd == "/help":
            await self._mount_message(UserMessage(command))
            help_body = (
                "Commands: /quit, /clear, /offload, /editor, /mcp, "
                "/model [--model-params JSON] [--default], /reload, "
                "/skill:<name>, /remember, /skill-creator, /theme, /tokens, "
                "/threads, /trace, "
                "/update, /auto-update, /changelog, /docs, /feedback, /help\n\n"
                "Interactive Features:\n"
                "  Enter           Submit your message\n"
                f"  {newline_shortcut():<15} Insert newline\n"
                "  Ctrl+X          Open prompt in external editor\n"
                "  Shift+Tab       Toggle auto-approve mode\n"
                "  @filename       Auto-complete files and inject content\n"
                "  /command        Slash commands (/help, /clear, /quit)\n"
                "  !command        Run shell commands directly\n\n"
                "Docs: "
            )
            help_text = Content.assemble(
                (help_body, "dim italic"),
                (DOCS_URL, TStyle(dim=True, italic=True, link=DOCS_URL)),
            )
            await self._mount_message(AppMessage(help_text))

        elif cmd in {"/changelog", "/docs", "/feedback"}:
            await self._open_url_command(command, cmd)
        elif cmd == "/version":
            await self._mount_message(UserMessage(command))
            # Show CLI and SDK package versions
            try:
                from deepagents_cli._version import (
                    __version__ as cli_version,
                )

                cli_line = f"deepagents-cli version: {cli_version}"
            except ImportError:
                logger.debug("deepagents_cli._version module not found")
                cli_line = "deepagents-cli version: unknown"
            except Exception:
                logger.warning("Unexpected error looking up CLI version", exc_info=True)
                cli_line = "deepagents-cli version: unknown"
            try:
                from importlib.metadata import (
                    PackageNotFoundError,
                    version as _pkg_version,
                )

                sdk_version = _pkg_version("deepagents")
                sdk_line = f"deepagents (SDK) version: {sdk_version}"
            except PackageNotFoundError:
                logger.debug("deepagents SDK package not found in environment")
                sdk_line = "deepagents (SDK) version: unknown"
            except Exception:
                logger.warning("Unexpected error looking up SDK version", exc_info=True)
                sdk_line = "deepagents (SDK) version: unknown"
            await self._mount_message(AppMessage(f"{cli_line}\n{sdk_line}"))
        elif cmd == "/clear":
            self._pending_messages.clear()
            self._queued_widgets.clear()
            await self._clear_messages()
            self._context_tokens = 0
            self._tokens_approximate = False
            self._update_tokens(0)
            # Clear status message (e.g., "Interrupted" from previous session)
            self._update_status("")
            # Reset thread to start fresh conversation
            if self._session_state:
                new_thread_id = self._session_state.reset_thread()
                try:
                    banner = self.query_one("#welcome-banner", WelcomeBanner)
                    banner.update_thread_id(new_thread_id)
                except NoMatches:
                    pass
                await self._mount_message(
                    AppMessage(f"Started new thread: {new_thread_id}")
                )
        elif cmd == "/editor":
            await self.action_open_editor()
        elif cmd in {"/offload", "/compact"}:
            await self._mount_message(UserMessage(command))
            await self._handle_offload()
        elif cmd == "/threads":
            await self._show_thread_selector()
        elif cmd == "/trace":
            await self._handle_trace_command(command)
        elif cmd == "/update":
            await self._handle_update_command()
        elif cmd == "/auto-update":
            await self._handle_auto_update_toggle()
        elif cmd == "/tokens":
            await self._mount_message(UserMessage(command))
            if self._context_tokens > 0:
                count = self._context_tokens
                formatted = format_token_count(count)

                model_name = settings.model_name
                context_limit = settings.model_context_limit

                if context_limit is not None:
                    limit_str = format_token_count(context_limit)
                    pct = count / context_limit * 100
                    usage = f"{formatted} / {limit_str} tokens ({pct:.0f}%)"
                else:
                    usage = f"{formatted} tokens used"

                msg = f"{usage} \u00b7 {model_name}" if model_name else usage

                conv_tokens = await self._get_conversation_token_count()
                if conv_tokens is not None:
                    overhead = max(0, count - conv_tokens)
                    overhead_str = format_token_count(overhead)
                    conv_str = format_token_count(conv_tokens)

                    overhead_unit = " tokens" if overhead < 1000 else ""  # noqa: PLR2004  # not bothersome, cosmetic
                    conv_unit = " tokens" if conv_tokens < 1000 else ""  # noqa: PLR2004  # not bothersome, cosmetic

                    msg += (
                        f"\n\u251c System prompt + tools: ~{overhead_str}{overhead_unit} (fixed)"  # noqa: E501
                        f"\n\u2514 Conversation: ~{conv_str}{conv_unit}"
                    )

                await self._mount_message(AppMessage(msg))
            else:
                model_name = settings.model_name
                context_limit = settings.model_context_limit

                parts: list[str] = ["No token usage yet"]
                if context_limit is not None:
                    limit_str = format_token_count(context_limit)
                    parts.append(f"{limit_str} token context window")
                if model_name:
                    parts.append(model_name)

                await self._mount_message(AppMessage(" · ".join(parts)))
        elif cmd == "/remember" or cmd.startswith("/remember "):
            # Convenience alias for /skill:remember — shorter and discoverable
            # before skill loading completes.
            args = command.strip()[len("/remember") :].strip()
            rewritten = f"/skill:remember {args}" if args else "/skill:remember"
            await self._handle_skill_command(rewritten)
        elif cmd == "/skill-creator" or cmd.startswith("/skill-creator "):
            # Convenience alias for /skill:skill-creator — shorter and
            # discoverable before skill loading completes.
            args = command.strip()[len("/skill-creator") :].strip()
            rewritten = (
                f"/skill:skill-creator {args}" if args else "/skill:skill-creator"
            )
            await self._handle_skill_command(rewritten)
        elif cmd == "/mcp":
            await self._show_mcp_viewer()
        elif cmd == "/theme":
            await self._show_theme_selector()
        elif cmd == "/model" or cmd.startswith("/model "):
            model_arg = None
            set_default = False
            extra_kwargs: dict[str, Any] | None = None
            if cmd.startswith("/model "):
                raw_arg = command.strip()[len("/model ") :].strip()
                try:
                    raw_arg, extra_kwargs = _extract_model_params_flag(raw_arg)
                except (ValueError, TypeError) as exc:
                    await self._mount_message(UserMessage(command))
                    await self._mount_message(ErrorMessage(str(exc)))
                    return
                if raw_arg.startswith("--default"):
                    set_default = True
                    model_arg = raw_arg[len("--default") :].strip() or None
                else:
                    model_arg = raw_arg or None

            if set_default:
                await self._mount_message(UserMessage(command))
                if extra_kwargs:
                    await self._mount_message(
                        ErrorMessage(
                            "--model-params cannot be used with --default. "
                            "Model params are applied per-session, not "
                            "persisted."
                        )
                    )
                elif model_arg == "--clear":
                    await self._clear_default_model()
                elif model_arg:
                    await self._set_default_model(model_arg)
                else:
                    await self._mount_message(
                        AppMessage(
                            "Usage: /model --default provider:model\n"
                            "       /model --default --clear"
                        )
                    )
            elif model_arg:
                # Direct switch: /model claude-sonnet-4-5
                await self._mount_message(UserMessage(command))
                await self._switch_model(model_arg, extra_kwargs=extra_kwargs)
            else:
                await self._show_model_selector(extra_kwargs=extra_kwargs)
        elif cmd == "/reload":
            await self._mount_message(UserMessage(command))
            try:
                changes = settings.reload_from_environment()

                from deepagents_cli.model_config import clear_caches

                clear_caches()
            except (OSError, ValueError):
                logger.exception("Failed to reload configuration")
                await self._mount_message(
                    AppMessage(
                        "Failed to reload configuration. Check your .env "
                        "file and environment variables for syntax errors, "
                        "then try again."
                    )
                )
                return

            # Reload user themes from config.toml and re-register with Textual
            theme_reload_ok = True
            try:
                theme.reload_registry()
                self._register_custom_themes()
            except Exception:
                theme_reload_ok = False
                logger.warning("Failed to reload user themes", exc_info=True)

            if changes:
                report = "Configuration reloaded. Changes:\n" + "\n".join(
                    f"  - {change}" for change in changes
                )
            else:
                report = "Configuration reloaded. No changes detected."
            report += "\nModel config caches cleared."
            if theme_reload_ok:
                report += "\nTheme registry reloaded."
            else:
                report += (
                    "\nTheme registry reload failed. Check config.toml for errors."
                )
            await self._mount_message(AppMessage(report))

            # Re-discover skills so autocomplete reflects any new/removed skills
            self.run_worker(
                self._discover_skills(),
                exclusive=True,
                group="startup-skill-discovery",
            )
        elif cmd.startswith("/skill:"):
            await self._handle_skill_command(command)
        else:
            await self._mount_message(UserMessage(command))
            await self._mount_message(AppMessage(f"Unknown command: {cmd}"))

        # Anchor to bottom so command output stays visible
        with suppress(NoMatches, ScreenStackError):
            self.query_one("#chat", VerticalScroll).anchor()

    async def _handle_skill_command(self, command: str) -> None:
        """스킬을 로드하고 호출하여 `/skill:<name>` 명령을 처리합니다.

        캐시된 메타데이터(시작 시 채워짐)에서 기술을 찾아 캐시 누락 시 새로운 파일 시스템 워크로 돌아갑니다. `SKILL.md` 본문을 읽고 이를
        사용자가 제공한 인수와 함께 프롬프트 봉투에 래핑한 다음 작성된 메시지를 에이전트에 보냅니다.

Args:
            command: 전체 명령 문자열(예: `/skill:web-research find X`).

        """
        from deepagents_cli.command_registry import parse_skill_command
        from deepagents_cli.skills.load import load_skill_content

        skill_name, args = parse_skill_command(command)
        if not skill_name:
            await self._mount_message(UserMessage(command))
            await self._mount_message(AppMessage("Usage: /skill:<name> [args]"))
            return

        # Fast path: look up from the cached discovery results
        cached = next(
            (s for s in self._discovered_skills if s["name"] == skill_name),
            None,
        )
        allowed_roots = self._skill_allowed_roots

        # Cache miss — fall back to fresh discovery (offloaded to thread)
        if cached is None:
            try:
                skills, allowed_roots = await asyncio.to_thread(
                    self._discover_skills_and_roots
                )
                # Backfill cache so subsequent invocations are fast
                self._discovered_skills = skills
                self._skill_allowed_roots = allowed_roots
                cached = next((s for s in skills if s["name"] == skill_name), None)
            except OSError as exc:
                logger.warning(
                    "Filesystem error loading skill %r", skill_name, exc_info=True
                )
                await self._mount_message(UserMessage(command))
                await self._mount_message(
                    AppMessage(
                        f"Could not load skill: {skill_name}. Filesystem error: {exc}"
                    )
                )
                return
            except Exception as exc:
                logger.warning(
                    "Error searching for skill %r", skill_name, exc_info=True
                )
                await self._mount_message(UserMessage(command))
                await self._mount_message(
                    AppMessage(
                        f"Error loading skill: {skill_name}. "
                        f"Unexpected error: {type(exc).__name__}: {exc}"
                    )
                )
                return

        if cached is None:
            await self._mount_message(UserMessage(command))
            await self._mount_message(AppMessage(f"Skill not found: {skill_name}"))
            return

        # Load SKILL.md content (filesystem I/O offloaded to thread)
        skill_path = cached["path"]

        def _load() -> str | None:
            return load_skill_content(str(skill_path), allowed_roots=allowed_roots)

        try:
            content = await asyncio.to_thread(_load)
        except PermissionError as exc:
            logger.warning(
                "Containment check failed for skill %r", skill_name, exc_info=True
            )
            await self._mount_message(UserMessage(command))
            await self._mount_message(AppMessage(str(exc)))
            return
        except OSError as exc:
            logger.warning(
                "Filesystem error loading skill %r", skill_name, exc_info=True
            )
            await self._mount_message(UserMessage(command))
            await self._mount_message(
                AppMessage(
                    f"Could not load skill: {skill_name}. Filesystem error: {exc}"
                )
            )
            return
        except Exception as exc:
            logger.warning("Error reading skill %r", skill_name, exc_info=True)
            await self._mount_message(UserMessage(command))
            await self._mount_message(
                AppMessage(
                    f"Error loading skill: {skill_name}. "
                    f"Unexpected error: {type(exc).__name__}: {exc}"
                )
            )
            return

        if content is None:
            await self._mount_message(UserMessage(command))
            await self._mount_message(
                AppMessage(
                    f"Could not read content for skill: {skill_name}. "
                    "Check that the SKILL.md file exists, is readable, "
                    "and is saved as UTF-8."
                )
            )
            return

        if not content.strip():
            await self._mount_message(UserMessage(command))
            await self._mount_message(
                AppMessage(
                    f"Skill '{skill_name}' has an empty SKILL.md file. "
                    "Add instructions to the file before invoking."
                )
            )
            return

        prompt = (
            f"I'm invoking the skill `{cached['name']}`. "
            "Below are the full instructions from the skill's SKILL.md file. "
            "Follow these instructions to complete the task.\n\n"
            f"---\n{content}\n---"
        )
        if args:
            prompt += f"\n\n**User request:** {args}"

        await self._mount_message(
            SkillMessage(
                skill_name=cached["name"],
                description=str(cached.get("description", "")),
                source=str(cached.get("source", "")),
                body=content,
                args=args,
            )
        )
        await self._send_to_agent(
            prompt,
            message_kwargs={
                "additional_kwargs": {
                    "__skill": {
                        "name": cached["name"],
                        "description": str(cached.get("description", "")),
                        "source": str(cached.get("source", "")),
                        "args": args,
                    },
                },
            },
        )

    async def _get_conversation_token_count(self) -> int | None:
        """대략적인 대화 전용 토큰 수를 반환합니다.

Returns:
            토큰 수는 정수로 표시되며, 상태를 사용할 수 없는 경우에는 `None`입니다.

        """
        if not self._agent:
            return None
        try:
            from langchain_core.messages.utils import (
                count_tokens_approximately,
            )

            config: RunnableConfig = {
                "configurable": {"thread_id": self._lc_thread_id},
            }
            state = await self._agent.aget_state(config)
            if not state or not state.values:
                return None
            messages = state.values.get("messages", [])
            if not messages:
                return None
            return count_tokens_approximately(messages)
        except Exception:  # best-effort for /tokens display
            logger.debug("Failed to retrieve conversation token count", exc_info=True)
            return None

    def _resolve_offload_budget_str(self) -> str | None:
        """오프로드 보존 예산을 사람이 읽을 수 있는 문자열로 확인합니다.

        모델을 인스턴스화하고 요약 기본값을 계산하므로 이는 사소한 접근자가 아닙니다.

Returns:
            `"20.0K (10% of 200.0K)"` 또는 `"last 6 messages"`과 같은 문자열입니다. 예산을 결정할 수 없는
            경우에는 `None`입니다.

        """
        from deepagents_cli.config import create_model, settings

        try:
            from deepagents.middleware.summarization import (
                compute_summarization_defaults,
            )

            model_spec = f"{settings.model_provider}:{settings.model_name}"
            result = create_model(
                model_spec,
                profile_overrides=self._profile_override,
            )
            defaults = compute_summarization_defaults(result.model)
            from deepagents_cli.offload import format_offload_limit

            return format_offload_limit(
                defaults["keep"],
                settings.model_context_limit,
            )
        except Exception:  # best-effort for /tokens display
            logger.debug("Failed to compute offload budget string", exc_info=True)
            return None

    async def _handle_offload(self) -> None:
        """오래된 메시지를 오프로드하여 컨텍스트 창 공간을 확보하세요."""
        from deepagents_cli.config import settings
        from deepagents_cli.offload import (
            OffloadModelError,
            OffloadThresholdNotMet,
            perform_offload,
        )

        if not self._agent or not self._lc_thread_id:
            await self._mount_message(
                AppMessage("Nothing to offload \u2014 start a conversation first")
            )
            return

        if self._agent_running:
            await self._mount_message(
                AppMessage("Cannot offload while agent is running")
            )
            return

        config: RunnableConfig = {"configurable": {"thread_id": self._lc_thread_id}}

        try:
            state_values = await self._get_thread_state_values(self._lc_thread_id)
        except Exception as exc:  # noqa: BLE001
            await self._mount_message(ErrorMessage(f"Failed to read state: {exc}"))
            return

        if not state_values:
            await self._mount_message(
                AppMessage("Nothing to offload \u2014 start a conversation first")
            )
            return

        # Prevent concurrent user input while offload modifies state
        self._agent_running = True
        try:
            from deepagents_cli.hooks import dispatch_hook

            await dispatch_hook("context.offload", {})
            # Keep old hook name for backward compatibility
            await dispatch_hook("context.compact", {})
            await self._set_spinner("Offloading")

            result = await perform_offload(
                messages=state_values.get("messages", []),
                prior_event=state_values.get("_summarization_event"),
                thread_id=self._lc_thread_id,
                model_spec=(f"{settings.model_provider}:{settings.model_name}"),
                profile_overrides=self._profile_override,
                context_limit=settings.model_context_limit,
                total_context_tokens=self._context_tokens,
                backend=self._backend,
            )

            if isinstance(result, OffloadThresholdNotMet):
                conv_str = format_token_count(result.conversation_tokens)
                if (
                    result.total_context_tokens > 0
                    and result.context_limit is not None
                    and result.total_context_tokens > result.context_limit
                ):
                    total_str = format_token_count(
                        result.total_context_tokens,
                    )
                    await self._mount_message(
                        AppMessage(
                            f"Offload threshold not met \u2014 conversation "
                            f"is only ~{conv_str} tokens.\n\n"
                            f"The remaining context "
                            f"({total_str} tokens) is system overhead "
                            f"that can't be offloaded.\n\n"
                            f"Use /tokens for a full breakdown."
                        )
                    )
                else:
                    await self._mount_message(
                        AppMessage(
                            f"Offload threshold not met \u2014 conversation "
                            f"(~{conv_str} tokens) is within the "
                            f"retention budget "
                            f"({result.budget_str}).\n\n"
                            f"Use /tokens for a full breakdown."
                        )
                    )
                return

            # OffloadResult — success
            if result.offload_warning:
                await self._mount_message(ErrorMessage(result.offload_warning))

            if remote := self._remote_agent():
                await remote.aensure_thread(config)  # ty: ignore[invalid-argument-type]

            await self._agent.aupdate_state(
                config, {"_summarization_event": result.new_event}
            )

            before = format_token_count(result.tokens_before)
            after = format_token_count(result.tokens_after)
            await self._mount_message(
                AppMessage(
                    f"Offloaded {result.messages_offloaded} older messages, "
                    f"freeing up context window space.\n"
                    f"Context: {before} \u2192 {after} tokens "
                    f"({result.pct_decrease}% decrease), "
                    f"{result.messages_kept} messages kept."
                )
            )

            self._on_tokens_update(result.tokens_after)
            from deepagents_cli.textual_adapter import _persist_context_tokens

            await _persist_context_tokens(self._agent, config, result.tokens_after)

        except OffloadModelError as exc:
            logger.warning("Offload model creation failed: %s", exc, exc_info=True)
            await self._mount_message(ErrorMessage(str(exc)))
        except Exception as exc:  # surface offload errors to user
            logger.exception("Offload failed")
            await self._mount_message(ErrorMessage(f"Offload failed: {exc}"))
        finally:
            self._agent_running = False
            try:
                await self._set_spinner(None)
            except Exception:  # best-effort spinner cleanup
                logger.exception("Failed to dismiss spinner after offload")

    async def _handle_user_message(self, message: str) -> None:
        """에이전트에 보낼 사용자 메시지를 처리합니다.

Args:
            message: 사용자의 메시지

        """
        # Mount the user message
        await self._mount_message(UserMessage(message))
        await self._send_to_agent(message)

    async def _send_to_agent(
        self,
        message: str,
        *,
        message_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """에이전트에 메시지를 보내고 실행을 시작합니다.

        이는 낮은 수준의 전송 경로입니다. 어떤 위젯도 마운트하지 않습니다. 호출자는 이 메서드를 호출하기 전에 적절한 시각적 표현(예:
        `UserMessage`, `SkillMessage`)을 마운트해야 합니다.

Args:
            message: 에이전트에게 보낼 프롬프트입니다.
            message_kwargs: 추가 필드가 스트림 입력 메시지 사전에 병합되었습니다(예: 기술 메타데이터의 경우
                            `additional_kwargs`).

        """
        # Anchor to bottom so streaming response stays visible
        with suppress(NoMatches, ScreenStackError):
            self.query_one("#chat", VerticalScroll).anchor()

        # Check if agent is available
        if self._agent and self._ui_adapter and self._session_state:
            self._agent_running = True

            if self._chat_input:
                self._chat_input.set_cursor_active(active=False)

            # Use run_worker to avoid blocking the main event loop
            # This allows the UI to remain responsive during agent execution
            self._agent_worker = self.run_worker(
                self._run_agent_task(message, message_kwargs=message_kwargs),
                exclusive=False,
            )
        else:
            await self._mount_message(
                AppMessage("Agent not configured for this session.")
            )

    async def _run_agent_task(
        self,
        message: str,
        *,
        message_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """백그라운드 작업자에서 에이전트 작업을 실행합니다.

        이는 텍스트 작업자에서 실행되므로 기본 이벤트 루프가 계속 응답합니다.

Args:
            message: 에이전트에게 보낼 프롬프트입니다.
            message_kwargs: 추가 필드가 스트림 입력 메시지 사전에 병합되었습니다(예: 기술 메타데이터의 경우
                            `additional_kwargs`).

        """
        # Caller ensures _ui_adapter is set (checked in _handle_user_message)
        if self._ui_adapter is None:
            return
        from deepagents_cli.textual_adapter import execute_task_textual

        # Create the stats object up-front and store on the app so
        # exit() can merge it synchronously if the worker is cancelled
        # before this method can return (e.g. Ctrl+D during HITL).
        turn_stats = SessionStats()
        self._inflight_turn_stats = turn_stats
        self._inflight_turn_start = time.monotonic()
        try:
            await execute_task_textual(
                user_input=message,
                agent=self._agent,
                assistant_id=self._assistant_id,
                session_state=self._session_state,
                adapter=self._ui_adapter,
                backend=self._backend,
                image_tracker=self._image_tracker,
                sandbox_type=self._sandbox_type,
                message_kwargs=message_kwargs,
                context=CLIContext(
                    model=self._model_override,
                    model_params=self._model_params_override or {},
                ),
                turn_stats=turn_stats,
            )
        except Exception as e:  # Resilient tool rendering
            logger.exception("Agent execution failed")
            # Ensure any in-flight tool calls don't remain stuck in "Running..."
            # when streaming aborts before tool results arrive.
            if self._ui_adapter:
                self._ui_adapter.finalize_pending_tools_with_error(f"Agent error: {e}")
            try:
                await self._mount_message(ErrorMessage(f"Agent error: {e}"))
            except Exception:
                logger.debug(
                    "Could not mount error message (app closing?)", exc_info=True
                )
        finally:
            # Merge turn stats before cleanup — _cleanup_agent_task may raise
            # during teardown (widget removal on a torn-down DOM), and stats
            # should ideally be captured regardless.
            # exit() clears _inflight_turn_stats when it merges, so
            # checking for None prevents double-counting.
            if self._inflight_turn_stats is not None:
                self._session_stats.merge(turn_stats)
                self._inflight_turn_stats = None
            await self._cleanup_agent_task()

    async def _process_next_from_queue(self) -> None:
        """큐에 다음 메시지가 있으면 처리합니다.

        FIFO 순서로 다음 보류 메시지를 대기열에서 빼고 처리합니다. 재진입 실행을 방지하기 위해 `_processing_pending` 플래그를
        사용합니다.

        """
        if self._processing_pending or not self._pending_messages or self._exit:
            return

        self._processing_pending = True
        try:
            msg = self._pending_messages.popleft()

            # Remove the ephemeral queued-message widget
            if self._queued_widgets:
                widget = self._queued_widgets.popleft()
                await widget.remove()

            await self._process_message(msg.text, msg.mode)
        except Exception:
            logger.exception("Failed to process queued message")
            await self._mount_message(
                ErrorMessage(f"Failed to process queued message: {msg.text[:60]}")
            )
        finally:
            self._processing_pending = False

        # Command mode messages complete synchronously without spawning
        # a worker, so cleanup won't fire again. Continue draining the
        # queue if no worker was started.
        busy = self._agent_running or self._shell_running
        if not busy and self._pending_messages:
            await self._process_next_from_queue()

    async def _cleanup_agent_task(self) -> None:
        """에이전트 작업이 완료되거나 취소된 후 정리합니다."""
        self._agent_running = False
        self._agent_worker = None

        # Remove spinner if present
        await self._set_spinner(None)

        if self._chat_input:
            self._chat_input.set_cursor_active(active=True)

        # Ensure token display is restored (in case of early cancellation).
        # Pass the cached approximate flag so an interrupted "+" isn't clobbered.
        self._show_tokens(approximate=self._tokens_approximate)

        try:
            await self._maybe_drain_deferred()
        except Exception:
            logger.exception("Failed to drain deferred actions during agent cleanup")
            with suppress(Exception):
                await self._mount_message(
                    ErrorMessage(
                        "A deferred action failed after task completion. "
                        "You may need to retry the operation."
                    )
                )

        # Process next message from queue if any
        await self._process_next_from_queue()

    @staticmethod
    def _convert_messages_to_data(messages: list[Any]) -> list[MessageData]:
        """LangChain 메시지를 가벼운 `MessageData` 개체로 변환합니다.

        이는 DOM 작업이 없는 순수 함수입니다. 도구 호출 일치는 여기에서 발생합니다. `ToolMessage` 결과는 `tool_call_id`과
        일치하고 해당 `MessageData`에 직접 저장됩니다.

Args:
            messages: 스레드 체크포인트의 LangChain 메시지 개체입니다.

Returns:
            `MessageStore.bulk_load`에 대한 `MessageData` 주문 목록이 준비되었습니다.

        """
        from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

        result: list[MessageData] = []
        # Maps tool_call_id -> index into result list
        pending_tool_indices: dict[str, int] = {}

        for msg in messages:
            if isinstance(msg, HumanMessage):
                content = (
                    msg.content if isinstance(msg.content, str) else str(msg.content)
                )
                if content.startswith("[SYSTEM]"):
                    continue

                # Detect skill invocations persisted via additional_kwargs
                skill_meta = (msg.additional_kwargs or {}).get("__skill")
                if isinstance(skill_meta, dict) and skill_meta.get("name"):
                    result.append(
                        MessageData(
                            type=MessageType.SKILL,
                            content="",
                            skill_name=skill_meta["name"],
                            skill_description=str(skill_meta.get("description", "")),
                            skill_source=str(skill_meta.get("source", "")),
                            skill_args=str(skill_meta.get("args", "")),
                            skill_body=content,
                        )
                    )
                else:
                    result.append(MessageData(type=MessageType.USER, content=content))

            elif isinstance(msg, AIMessage):
                # Extract text content
                content = msg.content
                text = ""
                if isinstance(content, str):
                    text = content.strip()
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text += block.get("text", "")
                        elif isinstance(block, str):
                            text += block
                    text = text.strip()

                if text:
                    result.append(MessageData(type=MessageType.ASSISTANT, content=text))

                # Track tool calls for later matching
                for tc in getattr(msg, "tool_calls", []):
                    tc_id = tc.get("id")
                    name = tc.get("name", "unknown")
                    args = tc.get("args", {})
                    data = MessageData(
                        type=MessageType.TOOL,
                        content="",
                        tool_name=name,
                        tool_args=args,
                        tool_status=ToolStatus.PENDING,
                    )
                    result.append(data)
                    if tc_id:
                        pending_tool_indices[tc_id] = len(result) - 1
                    else:
                        data.tool_status = ToolStatus.REJECTED

            elif isinstance(msg, ToolMessage):
                tc_id = getattr(msg, "tool_call_id", None)
                if tc_id and tc_id in pending_tool_indices:
                    idx = pending_tool_indices.pop(tc_id)
                    data = result[idx]
                    status = getattr(msg, "status", "success")
                    content = (
                        msg.content
                        if isinstance(msg.content, str)
                        else str(msg.content)
                    )
                    if status == "success":
                        data.tool_status = ToolStatus.SUCCESS
                    else:
                        data.tool_status = ToolStatus.ERROR
                    data.tool_output = content
                else:
                    logger.debug(
                        "ToolMessage with tool_call_id=%r could not be "
                        "matched to a pending tool call",
                        tc_id,
                    )

            else:
                logger.debug(
                    "Skipping unsupported message type %s during history conversion",
                    type(msg).__name__,
                )

        # Mark unmatched tool calls as rejected
        for idx in pending_tool_indices.values():
            result[idx].tool_status = ToolStatus.REJECTED

        return result

    async def _get_thread_state_values(self, thread_id: str) -> dict[str, Any]:
        """원격 체크포인터 폴백을 사용하여 스레드 상태 값을 가져옵니다.

        서버 모드에서 LangGraph 개발 서버는 디스크에 체크포인트가 있더라도 다시 시작한 후 빈 스레드 상태를 보고할 수 있습니다. 그런 일이
        발생하면 재개된 스레드가 계속 기록을 로드하고 올바르게 오프로드할 수 있도록 최신 체크포인트를 직접 읽으십시오.

Args:
            thread_id: 체크포인트 저장소에서 가져올 스레드 ID입니다.

Returns:
            채널 이름으로 키가 지정된 스레드 상태 값입니다. 빈 사전을 반환합니다.
                체크포인트된 값을 사용할 수 없는 경우.

        """
        if not self._agent:
            return {}

        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        state = await self._agent.aget_state(config)

        values: dict[str, Any] = {}
        if state and state.values:
            values = dict(state.values)

        messages = values.get("messages")
        if isinstance(messages, list) and messages:
            return values
        if not self._remote_agent():
            return values

        logger.debug(
            "Remote state empty for thread %s; falling back to local checkpointer",
            thread_id,
        )
        fallback_values = await self._read_channel_values_from_checkpointer(thread_id)
        fallback_messages = fallback_values.get("messages")
        if isinstance(fallback_messages, list) and fallback_messages:
            values["messages"] = fallback_messages
        if (
            values.get("_summarization_event") is None
            and "_summarization_event" in fallback_values
        ):
            values["_summarization_event"] = fallback_values["_summarization_event"]
        if (
            values.get("_context_tokens") is None
            and "_context_tokens" in fallback_values
        ):
            values["_context_tokens"] = fallback_values["_context_tokens"]
        return values

    async def _fetch_thread_history_data(self, thread_id: str) -> _ThreadHistoryPayload:
        """스레드에 대해 저장된 메시지를 가져오고 변환합니다.

        서버 모드에서 LangGraph 개발 서버는 빈 스레드 저장소로 시작하므로 HTTP API를 통한 `aget_state`은 디스크에 체크포인트가
        있어도 메시지를 반환하지 않습니다. 재개된 스레드가 기록을 로드하도록 보장하기 위해 SQLite 체크포인터를 직접 읽는 방식으로 돌아갑니다.

Args:
            thread_id: 체크포인트 저장소에서 가져올 스레드 ID입니다.

Returns:
            변환된 메시지 데이터와 지속되는 컨텍스트 토큰 수를 포함하는 페이로드입니다.

        """
        state_values = await self._get_thread_state_values(thread_id)
        raw_tokens = state_values.get("_context_tokens")
        context_tokens = (
            raw_tokens if isinstance(raw_tokens, int) and raw_tokens >= 0 else 0
        )
        messages = state_values.get("messages", [])

        if not messages:
            return _ThreadHistoryPayload([], context_tokens)

        # Server mode / direct checkpointer may return dicts; convert to
        # LangChain message objects so _convert_messages_to_data works.
        if messages and isinstance(messages[0], dict):
            from langchain_core.messages.utils import convert_to_messages

            messages = convert_to_messages(messages)

        # Offload conversion so large histories don't block the UI loop.
        data = await asyncio.to_thread(self._convert_messages_to_data, messages)
        return _ThreadHistoryPayload(data, context_tokens)

    @staticmethod
    async def _read_channel_values_from_checkpointer(thread_id: str) -> dict[str, Any]:
        """SQLite 체크포인터에서 직접 체크포인트 채널 값을 읽습니다.

Args:
            thread_id: 조회할 스레드 ID입니다.

Returns:
            최신 체크포인트의 채널 값 또는 빈 사전
                실패.

        """
        try:
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

            from deepagents_cli.sessions import get_db_path

            db_path = str(get_db_path())
            config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
            async with AsyncSqliteSaver.from_conn_string(db_path) as saver:
                tup = await saver.aget_tuple(config)
                if tup and tup.checkpoint:
                    channel_values = tup.checkpoint.get("channel_values", {})
                    if isinstance(channel_values, dict):
                        return dict(channel_values)
        except (ImportError, OSError) as exc:
            logger.warning(
                "Failed to read checkpointer directly for %s: %s",
                thread_id,
                exc,
            )
        except Exception:
            logger.warning(
                "Unexpected error reading checkpointer for %s",
                thread_id,
                exc_info=True,
            )
        return {}

    async def _upgrade_thread_message_link(
        self,
        widget: AppMessage,
        *,
        prefix: str,
        thread_id: str,
    ) -> None:
        """URL이 확인되면 일반 스레드 메시지를 연결된 메시지로 업그레이드합니다.

Args:
            widget: 이미 마운트된 앱 메시지입니다.
            prefix: 스레드 ID 앞의 텍스트 접두사입니다.
            thread_id: 해결할 스레드 ID입니다.

        """
        try:
            thread_msg = await self._build_thread_message(prefix, thread_id)
            if not isinstance(thread_msg, Content):
                logger.debug(
                    "Skipping thread link upgrade for %s: URL did not resolve",
                    thread_id,
                )
                return
            if widget.parent is None:
                logger.debug(
                    "Skipping thread link upgrade for %s: widget no longer mounted",
                    thread_id,
                )
                return
            # Keep serialized content in sync with the rendered content.
            widget._content = thread_msg
            widget.update(thread_msg)
        except Exception:
            logger.warning(
                "Failed to upgrade thread message link for %s",
                thread_id,
                exc_info=True,
            )

    def _schedule_thread_message_link(
        self,
        widget: AppMessage,
        *,
        prefix: str,
        thread_id: str,
    ) -> None:
        """스레드 URL 링크 확인을 예약하고 백그라운드에서 업데이트를 적용합니다.

Args:
            widget: 업데이트할 메시지 위젯입니다.
            prefix: 스레드 ID 앞의 텍스트 접두사입니다.
            thread_id: 해결할 스레드 ID입니다.

        """
        self.run_worker(
            self._upgrade_thread_message_link(
                widget,
                prefix=prefix,
                thread_id=thread_id,
            ),
            exclusive=False,
        )

    async def _load_thread_history(
        self,
        *,
        thread_id: str | None = None,
        preloaded_payload: _ThreadHistoryPayload | None = None,
    ) -> None:
        """스레드를 재개할 때 메시지 기록을 로드하고 렌더링합니다.

        `preloaded_payload`이 제공되면(예: `_resume_thread`에서) 해당 데이터를 재사용합니다. 그렇지 않으면 에이전트에서
        체크포인트 상태를 가져오고 저장된 메시지를 경량 `MessageData` 개체로 변환합니다. 그런 다음 이 메서드는 `MessageStore`에
        대량 로드되고 마지막 `WINDOW_SIZE` 위젯만 마운트하여 대규모 스레드에서 DOM 작업을 줄입니다.

Args:
            thread_id: 로드할 선택적 명시적 스레드 ID입니다.

                기본값은 현재입니다.
            preloaded_payload: 스레드에 대해 선택적으로 미리 가져온 기록 페이로드입니다.

        """
        history_thread_id = thread_id or self._lc_thread_id
        if not history_thread_id:
            logger.debug("Skipping history load: no thread ID available")
            return
        if preloaded_payload is None and not self._agent:
            logger.debug(
                "Skipping history load for %s: no active agent and no preloaded data",
                history_thread_id,
            )
            return

        try:
            # Fetch + convert, or reuse preloaded payload on thread switch.
            payload = (
                preloaded_payload
                if preloaded_payload is not None
                else await self._fetch_thread_history_data(history_thread_id)
            )
            if not payload.messages:
                return

            # Seed token cache from persisted state
            if payload.context_tokens > 0:
                self._on_tokens_update(payload.context_tokens)

            # 3. Bulk load into store (sets visible window)
            _archived, visible = self._message_store.bulk_load(payload.messages)

            # 5. Cache container ref (single query)
            try:
                messages_container = self.query_one("#messages", Container)
            except NoMatches:
                return

            # 6-7. Create and mount only visible widgets (max WINDOW_SIZE)
            widgets = [msg_data.to_widget() for msg_data in visible]
            if widgets:
                await messages_container.mount(*widgets)

            # 8. Render content for AssistantMessage after mount
            assistant_updates = [
                widget.set_content(msg_data.content)
                for widget, msg_data in zip(widgets, visible, strict=False)
                if isinstance(widget, AssistantMessage) and msg_data.content
            ]
            if assistant_updates:
                assistant_results = await asyncio.gather(
                    *assistant_updates,
                    return_exceptions=True,
                )
                for error in assistant_results:
                    if isinstance(error, Exception):
                        logger.warning(
                            "Failed to render assistant history message for %s: %s",
                            history_thread_id,
                            error,
                        )

            # 9. Add footer immediately and resolve link asynchronously
            thread_msg_widget = AppMessage(f"Resumed thread: {history_thread_id}")
            await self._mount_message(thread_msg_widget)
            self._schedule_thread_message_link(
                thread_msg_widget,
                prefix="Resumed thread",
                thread_id=history_thread_id,
            )

            # 10. Scroll once to bottom after history loads
            def scroll_to_end() -> None:
                with suppress(NoMatches):
                    chat = self.query_one("#chat", VerticalScroll)
                    chat.scroll_end(animate=False, immediate=True)

            self.set_timer(0.1, scroll_to_end)

        except Exception as e:  # Resilient history loading
            logger.exception(
                "Failed to load thread history for %s",
                history_thread_id,
            )
            await self._mount_message(AppMessage(f"Could not load history: {e}"))

    async def _mount_message(
        self, widget: Static | AssistantMessage | ToolCallMessage | SkillMessage
    ) -> None:
        """메시지 위젯을 메시지 영역에 마운트합니다.

        이 방법은 또한 메시지 데이터를 저장하고 위젯 수가 최대값을 초과할 때 정리를 처리합니다.

        ``#messages`` container is not present (e.g. the screen has been torn down
        during an interruption), the call is silently skipped to avoid cascading
        `NoMatches` 오류가 발생한 경우.

Args:
            widget: 마운트할 메시지 위젯

        """
        try:
            messages = self.query_one("#messages", Container)
        except NoMatches:
            return

        # During shutdown (e.g. Ctrl+D mid-stream) the container may still
        # be in the DOM tree but already detached, so mount() would raise
        # MountError. Bail out silently — the app is exiting anyway.
        if not messages.is_attached:
            return

        # Store message data for virtualization
        message_data = MessageData.from_widget(widget)
        # Ensure the widget's DOM id matches the store id so that
        # features like click-to-show-timestamp can look it up.
        if not widget.id:
            widget.id = message_data.id
        self._message_store.append(message_data)

        # Queued-message widgets must always stay at the bottom so they
        # remain visually anchored below the current agent response.
        if isinstance(widget, QueuedUserMessage):
            await messages.mount(widget)
        else:
            await self._mount_before_queued(messages, widget)

        # Prune old widgets if window exceeded
        await self._prune_old_messages()

        # Scroll to keep input bar visible
        try:
            input_container = self.query_one("#bottom-app-container", Container)
            input_container.scroll_visible()
        except NoMatches:
            pass

    async def _prune_old_messages(self) -> None:
        """창 크기를 초과하면 가장 오래된 메시지 위젯을 정리합니다.

        이렇게 하면 DOM에서 위젯이 제거되지만 위로 스크롤할 때 잠재적인 재수화를 위해 MessageStore에 데이터가 유지됩니다.

        """
        if not self._message_store.window_exceeded():
            return

        try:
            messages_container = self.query_one("#messages", Container)
        except NoMatches:
            logger.debug("Skipping pruning: #messages container not found")
            return

        to_prune = self._message_store.get_messages_to_prune()
        if not to_prune:
            return

        pruned_ids: list[str] = []
        for msg_data in to_prune:
            try:
                widget = messages_container.query_one(f"#{msg_data.id}")
                await widget.remove()
                pruned_ids.append(msg_data.id)
            except NoMatches:
                # Widget not found -- do NOT mark as pruned to avoid
                # desyncing the store from the actual DOM state
                logger.debug(
                    "Widget %s not found during pruning, skipping",
                    msg_data.id,
                )

        if pruned_ids:
            self._message_store.mark_pruned(pruned_ids)

    def _set_active_message(self, message_id: str | None) -> None:
        """활성 스트리밍 메시지를 설정합니다(정리되지 않음).

Args:
            message_id: 활성 메시지의 ID 또는 삭제할 경우 None입니다.

        """
        self._message_store.set_active_message(message_id)

    def _sync_message_content(self, message_id: str, content: str) -> None:
        """스트리밍 후 최종 메시지 콘텐츠를 스토어에 다시 동기화하세요.

        스트리밍이 완료되면 호출되어 저장소에 마운트 시 캡처된 빈 문자열 대신 전체 텍스트가 보관됩니다.

Args:
            message_id: 업데이트할 메시지의 ID입니다.
            content: 스트리밍 후 최종 콘텐츠입니다.

        """
        self._message_store.update_message(
            message_id,
            content=content,
            is_streaming=False,
        )

    async def _clear_messages(self) -> None:
        """메시지 영역과 메시지 저장소를 지웁니다."""
        # Clear the message store first
        self._message_store.clear()
        try:
            messages = self.query_one("#messages", Container)
            await messages.remove_children()
        except NoMatches:
            logger.warning(
                "Messages container (#messages) not found during clear; "
                "UI may be out of sync with message store"
            )

    def _pop_last_queued_message(self) -> None:
        """가장 최근에 대기 중인 메시지(LIFO)를 제거합니다.

        채팅 입력이 비어 있으면 제거된 텍스트가 복원되어 사용자가 편집하고 다시 제출할 수 있습니다. 그렇지 않으면 메시지가 삭제됩니다. 토스트
        메시지는 두 가지 결과를 구별합니다.

        호출자는 `_pending_messages`이 비어 있지 않은지 확인해야 합니다. 비동기 TOCTOU 경주의 경우 방어 가드가 포함됩니다.

        """
        if not self._pending_messages:
            return
        msg = self._pending_messages.pop()
        if self._queued_widgets:
            widget = self._queued_widgets.pop()
            widget.remove()
        else:
            logger.warning(
                "Queued-widget deque empty while pending-messages was not; "
                "widget/message tracking may be out of sync"
            )

        if not self._chat_input:
            logger.warning(
                "Chat input unavailable during queue pop; "
                "message text cannot be restored: %s",
                msg.text[:60],
            )
            self.notify("Queued message discarded", timeout=2)
            return

        if not self._chat_input.value.strip():
            self._chat_input.value = msg.text
            self.notify("Queued message moved to input", timeout=2)
        else:
            self.notify("Queued message discarded (input not empty)", timeout=3)

    def _discard_queue(self) -> None:
        """보류 중인 메시지, 지연된 작업 및 대기 중인 위젯을 지웁니다."""
        self._pending_messages.clear()
        for w in self._queued_widgets:
            w.remove()
        self._queued_widgets.clear()
        self._deferred_actions.clear()

    def _defer_action(self, action: DeferredAction) -> None:
        """동일한 종류의 기존 작업을 대체하여 지연된 작업을 대기열에 넣습니다.

        마지막 쓰기 우선: 사용자가 바쁜 동안 모델을 두 번 선택하면 최종 선택만 실행됩니다.

Args:
            action: 대기열에 대한 지연된 작업입니다.

        """
        self._deferred_actions = [
            a for a in self._deferred_actions if a.kind != action.kind
        ]
        self._deferred_actions.append(action)

    async def _maybe_drain_deferred(self) -> None:
        """서버 연결이 아직 진행 중이지 않은 경우 지연된 작업을 비웁니다."""
        if not self._connecting:
            await self._drain_deferred_actions()

    async def _drain_deferred_actions(self) -> None:
        """바쁜 동안 대기 중인 지연된 작업을 실행합니다(예: 모델/스레드 전환)."""
        while self._deferred_actions:
            action = self._deferred_actions.pop(0)
            try:
                await action.execute()
            except Exception:
                logger.exception(
                    "Failed to execute deferred action %r (callable=%r)",
                    action.kind,
                    action.execute,
                )
                label = action.kind.replace("_", " ")
                with suppress(Exception):
                    await self._mount_message(
                        ErrorMessage(
                            f"Deferred {label} failed unexpectedly. "
                            "You may need to retry the operation."
                        )
                    )

    def _cancel_worker(self, worker: Worker[None] | None) -> None:
        """메시지 대기열을 삭제하고 활성 작업자를 취소합니다.

Args:
            worker: 취소할 작업자입니다.

        """
        self._discard_queue()
        if worker is not None:
            worker.cancel()

    def action_quit_or_interrupt(self) -> None:
        """Ctrl+C 처리 - 에이전트 중단, 승인 거부 또는 두 번 누르기 종료.

        우선 순위: 1. 쉘 명령이 실행 중인 경우 종료 2. 승인 메뉴가 활성화된 경우 거부 3. 에이전트가 실행 중인 경우 중단(입력 유지) 4.
        두 번 누르면(quit_pending) 종료 5. 그렇지 않으면 종료 힌트 표시

        """
        # If shell command is running, cancel the worker
        if self._shell_running and self._shell_worker:
            self._cancel_worker(self._shell_worker)
            self._quit_pending = False
            return

        # If approval menu is active, reject it before cancelling the agent worker.
        # During HITL the agent worker remains active while awaiting approval,
        # so this must be checked before the worker cancellation branch to
        # avoid leaving a stale approval widget interactive after interruption.
        if self._pending_approval_widget:
            self._pending_approval_widget.action_select_reject()
            self._quit_pending = False
            return

        # If ask_user menu is active, cancel it before cancelling the agent
        # worker, following the same pattern as the approval widget above.
        if self._pending_ask_user_widget:
            self._pending_ask_user_widget.action_cancel()
            self._quit_pending = False
            return

        # If agent is running, interrupt it and discard queued messages
        if self._agent_running and self._agent_worker:
            self._cancel_worker(self._agent_worker)
            self._quit_pending = False
            return

        # Double Ctrl+C to quit
        if self._quit_pending:
            self.exit()
        else:
            self._arm_quit_pending("Ctrl+C")

    def _arm_quit_pending(self, shortcut: str) -> None:
        """보류 중인 종료 플래그를 설정하고 일치하는 힌트를 표시합니다.

Args:
            shortcut: 종료 힌트에 표시되는 키 코드입니다.

        """
        self._quit_pending = True
        quit_timeout = 3
        self.notify(
            f"Press {shortcut} again to quit", timeout=quit_timeout, markup=False
        )
        self.set_timer(quit_timeout, lambda: setattr(self, "_quit_pending", False))

    def action_interrupt(self) -> None:
        """이스케이프 키를 처리합니다.

        우선순위: 1. 모달 화면이 활성화되어 있으면 해제 2. 완료 팝업이 열려 있으면 해제 3. 입력이 명령/셸 모드인 경우 일반 모드로 종료 4.
        셸 명령이 실행 중이면 종료 5. 승인 메뉴가 활성화되어 있으면 거부 6. 사용자에게 묻기 메뉴가 활성화되어 있으면 취소 7. 대기 중인
        메시지가 있으면 마지막 메시지 팝(LIFO) 8. 에이전트가 실행 중이면 중단

        """
        from deepagents_cli.widgets.thread_selector import ThreadSelectorScreen

        if (
            isinstance(self.screen, ThreadSelectorScreen)
            and self.screen.is_delete_confirmation_open
        ):
            self.screen.action_cancel()
            return

        # If a modal screen is active, let it cancel itself (so it can
        # restore state, e.g. the theme selector reverts the previewed theme).
        # Fall back to a plain dismiss for modals without action_cancel.
        if isinstance(self.screen, ModalScreen):
            cancel = getattr(self.screen, "action_cancel", None)
            if cancel is not None:
                cancel()
            else:
                self.screen.dismiss(None)
            return

        # Close completion popup or exit slash/shell command mode
        if self._chat_input:
            if self._chat_input.dismiss_completion():
                return
            if self._chat_input.exit_mode():
                return

        # If shell command is running, cancel the worker
        if self._shell_running and self._shell_worker:
            self._cancel_worker(self._shell_worker)
            return

        # If approval menu is active, reject it before cancelling the agent worker.
        # During HITL the agent worker remains active while awaiting approval,
        # so this must be checked before the worker cancellation branch to
        # avoid leaving a stale approval widget interactive after interruption.
        if self._pending_approval_widget:
            self._pending_approval_widget.action_select_reject()
            return

        # If ask_user menu is active, cancel it before cancelling the agent
        # worker, following the same pattern as the approval widget above.
        if self._pending_ask_user_widget:
            self._pending_ask_user_widget.action_cancel()
            return

        # If queued messages exist, pop the last one (LIFO) instead of
        # interrupting the agent.  This lets the user retract queued messages
        # one at a time; once the queue is empty the next ESC will interrupt.
        if self._pending_messages:
            self._pop_last_queued_message()
            return

        # If agent is running, interrupt it and discard queued messages
        if self._agent_running and self._agent_worker:
            self._cancel_worker(self._agent_worker)
            return

    def action_quit_app(self) -> None:
        """종료 작업을 처리합니다(Ctrl+D)."""
        from deepagents_cli.widgets.thread_selector import (
            DeleteThreadConfirmScreen,
            ThreadSelectorScreen,
        )

        if isinstance(self.screen, ThreadSelectorScreen):
            self.screen.action_delete_thread()
            return
        if isinstance(self.screen, DeleteThreadConfirmScreen):
            if self._quit_pending:
                self.exit()
                return
            self._arm_quit_pending("Ctrl+D")
            return
        self.exit()

    def exit(
        self,
        result: Any = None,  # noqa: ANN401  # Dynamic LangGraph stream result type
        return_code: int = 0,
        message: Any = None,  # noqa: ANN401  # Dynamic LangGraph message type
    ) -> None:
        """해당하는 경우 앱을 종료하고 iTerm2 커서 가이드를 복원합니다.

        Textual을 정리하기 전에 iTerm2의 커서 안내를 복원하기 위해 상위 항목을 재정의합니다. atexit 핸들러는 비정상적인 종료에 대한
        대체 역할을 합니다.

Args:
            result: 앱 실행기에 전달된 반환 값입니다.
            return_code: 종료 코드(오류의 경우 0이 아님)
            message: 종료 시 표시할 선택적 메시지입니다.

        """
        # Merge in-flight turn stats before any cleanup that might raise.
        # When the agent worker is cancelled (e.g. Ctrl+D during a pending tool
        # call), the worker's finally block will see _inflight_turn_stats is
        # already None and skip the merge.
        inflight = self._inflight_turn_stats
        if inflight is not None:
            self._inflight_turn_stats = None
            if not inflight.wall_time_seconds:
                inflight.wall_time_seconds = (
                    time.monotonic() - self._inflight_turn_start
                )
            self._session_stats.merge(inflight)

        # Discard queued messages so _cleanup_agent_task won't try to
        # process them after the event loop is torn down, and cancel
        # active workers so their subprocesses are terminated
        # (SIGTERM → SIGKILL) instead of being orphaned.
        self._discard_queue()

        if self._shell_running and self._shell_worker:
            self._shell_worker.cancel()
        if self._agent_running and self._agent_worker:
            self._agent_worker.cancel()

        # Dispatch synchronously — the event loop is about to be torn down by
        # super().exit(), so an async task would never complete.
        from deepagents_cli.hooks import _dispatch_hook_sync, _load_hooks

        hooks = _load_hooks()
        if hooks:
            payload = json.dumps(
                {
                    "event": "session.end",
                    "thread_id": getattr(self, "_lc_thread_id", ""),
                }
            ).encode()
            _dispatch_hook_sync("session.end", payload, hooks)

        _write_iterm_escape(_ITERM_CURSOR_GUIDE_ON)
        super().exit(result=result, return_code=return_code, message=message)

    def action_toggle_auto_approve(self) -> None:
        """현재 세션에 대한 자동 승인 모드를 전환합니다.

        활성화되면 모든 도구 호출(셸 실행, 파일 쓰기/편집, 웹 검색, URL 가져오기)이 메시지 없이 실행됩니다. 상태 표시줄 표시기와 세션 상태를
        업데이트합니다.

        """
        from deepagents_cli.widgets.thread_selector import ThreadSelectorScreen

        if isinstance(self.screen, ThreadSelectorScreen):
            self.screen.action_focus_previous_filter()
            return
        # shift+tab is reused for navigation inside modal screens (e.g.
        # ModelSelectorScreen); skip the toggle so it doesn't fire through.
        if isinstance(self.screen, ModalScreen):
            return
        # Delegate shift+tab to ask_user navigation when interview is active.
        if self._pending_ask_user_widget is not None:
            self._pending_ask_user_widget.action_previous_question()
            return
        self._auto_approve = not self._auto_approve
        if self._status_bar:
            self._status_bar.set_auto_approve(enabled=self._auto_approve)
        if self._session_state:
            self._session_state.auto_approve = self._auto_approve

    def action_toggle_tool_output(self) -> None:
        """최신 도구 출력 또는 기술 본문의 확장/축소를 전환합니다."""
        # Try skill messages first (most recent collapsible content)
        with suppress(NoMatches):
            skill_messages = list(self.query(SkillMessage))
            for skill_msg in reversed(skill_messages):
                if skill_msg._stripped_body.strip():
                    skill_msg.toggle_body()
                    return
        # Fall back to tool messages with output
        with suppress(NoMatches):
            tool_messages = list(self.query(ToolCallMessage))
            for tool_msg in reversed(tool_messages):
                if tool_msg.has_output:
                    tool_msg.toggle_output()
                    return

    # Approval menu action handlers (delegated from App-level bindings)
    # NOTE: These only activate when approval widget is pending
    # AND input is not focused
    def action_approval_up(self) -> None:
        """승인 메뉴에서 위쪽 화살표를 처리합니다."""
        # Only handle if approval is active
        # (input handles its own up for history/completion)
        if self._pending_approval_widget and not self._is_input_focused():
            self._pending_approval_widget.action_move_up()

    def action_approval_down(self) -> None:
        """승인 메뉴에서 아래쪽 화살표를 처리합니다."""
        if self._pending_approval_widget and not self._is_input_focused():
            self._pending_approval_widget.action_move_down()

    def action_approval_select(self) -> None:
        """승인 메뉴에서 입력을 처리합니다."""
        # Only handle if approval is active AND input is not focused
        if self._pending_approval_widget and not self._is_input_focused():
            self._pending_approval_widget.action_select()

    def _is_input_focused(self) -> bool:
        """채팅 입력(또는 해당 텍스트 영역)에 포커스가 있는지 확인하세요.

Returns:
            입력 위젯에 포커스가 있으면 True이고, 그렇지 않으면 False입니다.

        """
        if not self._chat_input:
            return False
        focused = self.focused
        if focused is None:
            return False
        # Check if focused widget is the text area inside chat input
        return focused.id == "chat-input" or focused in self._chat_input.walk_children()

    def action_approval_yes(self) -> None:
        """승인 메뉴에서 yes/1을 처리합니다."""
        if self._pending_approval_widget:
            self._pending_approval_widget.action_select_approve()

    def action_approval_auto(self) -> None:
        """승인 메뉴에서 auto/2를 처리합니다."""
        if self._pending_approval_widget:
            self._pending_approval_widget.action_select_auto()

    def action_approval_no(self) -> None:
        """승인 메뉴에서 3번을 처리합니다."""
        if self._pending_approval_widget:
            self._pending_approval_widget.action_select_reject()

    def action_approval_escape(self) -> None:
        """승인 메뉴에서 이스케이프 처리 - 거부."""
        if self._pending_approval_widget:
            self._pending_approval_widget.action_select_reject()

    async def action_open_editor(self) -> None:
        """외부 편집기($VISUAL/$EDITOR)에서 현재 프롬프트 텍스트를 엽니다."""
        from deepagents_cli.editor import open_in_editor

        chat_input = self._chat_input
        if not chat_input or not chat_input._text_area:
            return

        current_text = chat_input._text_area.text or ""

        edited: str | None = None
        try:
            with self.suspend():
                edited = open_in_editor(current_text)
        except Exception:
            logger.warning("External editor failed", exc_info=True)
            self.notify(
                "External editor failed. Check $VISUAL/$EDITOR.",
                severity="error",
                timeout=5,
            )
            chat_input.focus_input()
            return

        if edited is not None:
            chat_input._text_area.text = edited
            lines = edited.split("\n")
            chat_input._text_area.move_cursor((len(lines) - 1, len(lines[-1])))
        chat_input.focus_input()

    def on_paste(self, event: Paste) -> None:
        """드래그/드롭 안정성을 위해 집중되지 않은 붙여넣기 이벤트를 채팅 입력으로 라우팅합니다."""
        if not self._chat_input:
            return
        if (
            self._pending_approval_widget
            or self._pending_ask_user_widget
            or self._is_input_focused()
        ):
            return
        if self._chat_input.handle_external_paste(event.text):
            event.prevent_default()
            event.stop()

    def on_app_focus(self) -> None:
        """터미널이 OS 포커스를 다시 얻으면 채팅 입력 포커스를 복원합니다.

        사용자가 `webbrowser.open`을 통해 링크를 열면 OS 포커스가 브라우저로 이동합니다. 터미널로 돌아오면 Textual은
        `AppFocus`을 실행합니다(FocusIn 이벤트를 지원하는 터미널 필요). 여기에서 채팅 입력에 다시 초점을 맞추면 입력할 준비가
        유지됩니다.

        """
        if not self._chat_input:
            return
        if isinstance(self.screen, ModalScreen):
            return
        if self._pending_approval_widget or self._pending_ask_user_widget:
            return
        self._chat_input.focus_input()

    def on_click(self, _event: Click) -> None:
        """명령줄에 집중하려면 터미널 어디에서나 클릭을 처리하세요."""
        if not self._chat_input:
            return
        # Don't steal focus from approval or ask_user widgets
        if self._pending_approval_widget or self._pending_ask_user_widget:
            return
        self.call_after_refresh(self._chat_input.focus_input)

    def on_mouse_up(self, event: MouseUp) -> None:  # noqa: ARG002  # Textual event handler signature
        """마우스를 놓으면 선택 항목이 클립보드에 복사됩니다."""
        from deepagents_cli.clipboard import copy_selection_to_clipboard

        copy_selection_to_clipboard(self)

    # =========================================================================
    # Model Switching
    # =========================================================================

    async def _show_model_selector(
        self,
        *,
        extra_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """대화형 모델 선택기를 모달 화면으로 표시합니다.

Args:
            extra_kwargs: `--model-params`의 추가 생성자 kwargs.

        """
        from functools import partial

        from deepagents_cli.config import settings
        from deepagents_cli.widgets.model_selector import ModelSelectorScreen

        def handle_result(result: tuple[str, str] | None) -> None:
            """모델 선택기 결과를 처리합니다."""
            if result is not None:
                model_spec, _ = result
                if self._agent_running or self._shell_running or self._connecting:
                    self._defer_action(
                        DeferredAction(
                            kind="model_switch",
                            execute=partial(
                                self._switch_model,
                                model_spec,
                                extra_kwargs=extra_kwargs,
                            ),
                        )
                    )
                    self.notify(
                        "Model will switch after current task completes.", timeout=3
                    )
                else:
                    self.call_later(
                        partial(
                            self._switch_model,
                            model_spec,
                            extra_kwargs=extra_kwargs,
                        )
                    )
            # Refocus input after modal closes
            if self._chat_input:
                self._chat_input.focus_input()

        screen = ModelSelectorScreen(
            current_model=settings.model_name,
            current_provider=settings.model_provider,
            cli_profile_override=self._profile_override,
        )
        self.push_screen(screen, handle_result)

    def _register_custom_themes(self) -> None:
        """모든 사용자 정의 테마(내장 LC + 사용자 정의)를 Textual에 등록합니다."""
        for name, entry in theme.ThemeEntry.REGISTRY.items():
            if entry.custom:
                c = entry.colors
                try:
                    self.register_theme(
                        Theme(
                            name=name,
                            primary=c.primary,
                            secondary=c.secondary,
                            accent=c.accent,
                            foreground=c.foreground,
                            background=c.background,
                            surface=c.surface,
                            panel=c.panel,
                            warning=c.warning,
                            error=c.error,
                            success=c.success,
                            dark=entry.dark,
                            variables={
                                "footer-key-foreground": c.primary,
                            },
                        )
                    )
                except Exception:
                    logger.warning(
                        "Failed to register theme '%s'; skipping",
                        name,
                        exc_info=True,
                    )

    async def _show_theme_selector(self) -> None:
        """대화형 테마 선택기를 모달 화면으로 표시합니다."""
        from deepagents_cli.widgets.theme_selector import ThemeSelectorScreen

        # Capture scroll state.  The submit handler may have already caused
        # a reflow that re-anchored to the bottom, so we save the *current*
        # offset and release the anchor to prevent further drift while the
        # modal is open.
        chat = self.query_one("#chat", VerticalScroll)
        saved_y = chat.scroll_y
        was_anchored = chat.is_anchored
        chat.release_anchor()

        def handle_result(result: str | None) -> None:
            """테마 선택기 결과를 처리합니다."""
            if result is not None:
                self.theme = result
                self.refresh_css(animate=False)

                async def _persist() -> None:
                    try:
                        ok = await asyncio.to_thread(save_theme_preference, result)
                        if not ok:
                            self.notify(
                                "Theme applied for this session but could not"
                                " be saved. Check logs for details.",
                                severity="warning",
                                timeout=6,
                                markup=False,
                            )
                    except Exception:
                        logger.warning(
                            "Failed to persist theme preference",
                            exc_info=True,
                        )
                        self.notify(
                            "Theme applied for this session but could not"
                            " be saved. Check logs for details.",
                            severity="warning",
                            timeout=6,
                            markup=False,
                        )

                self.call_later(_persist)
            # Restore scroll position, then re-anchor if it was anchored.
            chat.scroll_to(y=saved_y, animate=False)
            if was_anchored:
                chat.anchor()
            if self._chat_input:
                self._chat_input.focus_input()

        screen = ThemeSelectorScreen(current_theme=self.theme)
        self.push_screen(screen, handle_result)

    async def _show_mcp_viewer(self) -> None:
        """읽기 전용 MCP 서버/도구 뷰어를 모달 화면으로 표시합니다."""
        from deepagents_cli.widgets.mcp_viewer import MCPViewerScreen

        screen = MCPViewerScreen(server_info=self._mcp_server_info or [])

        def handle_result(result: None) -> None:  # noqa: ARG001
            if self._chat_input:
                self._chat_input.focus_input()

        self.push_screen(screen, handle_result)

    async def _show_thread_selector(self) -> None:
        """대화형 스레드 선택기를 모달 화면으로 표시합니다."""
        from functools import partial

        from deepagents_cli.sessions import get_cached_threads, get_thread_limit
        from deepagents_cli.widgets.thread_selector import ThreadSelectorScreen

        current = self._session_state.thread_id if self._session_state else None
        thread_limit = get_thread_limit()

        initial_threads = get_cached_threads(limit=thread_limit)

        def handle_result(result: str | None) -> None:
            """스레드 선택기 결과를 처리합니다."""
            if result is not None:
                if self._agent_running or self._shell_running or self._connecting:
                    self._defer_action(
                        DeferredAction(
                            kind="thread_switch",
                            execute=partial(self._resume_thread, result),
                        )
                    )
                    self.notify(
                        "Thread will switch after current task completes.", timeout=3
                    )
                else:
                    self.call_later(self._resume_thread, result)
            if self._chat_input:
                self._chat_input.focus_input()

        screen = ThreadSelectorScreen(
            current_thread=current,
            thread_limit=thread_limit,
            initial_threads=initial_threads,
        )
        self.push_screen(screen, handle_result)

    def _update_welcome_banner(
        self,
        thread_id: str,
        *,
        missing_message: str,
        warn_if_missing: bool,
    ) -> None:
        """배너가 탑재되면 환영 배너 스레드 ID를 업데이트합니다.

Args:
            thread_id: 배너에 표시할 스레드 ID입니다.
            missing_message: 배너가 누락된 경우 로그 메시지 템플릿입니다.
            warn_if_missing: 경고 수준에서 배너 누락 사례를 기록할지 여부입니다.

        """
        try:
            banner = self.query_one("#welcome-banner", WelcomeBanner)
            banner.update_thread_id(thread_id)
        except NoMatches:
            if warn_if_missing:
                logger.warning(missing_message, thread_id)
            else:
                logger.debug(missing_message, thread_id)

    async def _resume_thread(self, thread_id: str) -> None:
        """이전에 저장한 스레드를 재개합니다.

        선택한 스레드 기록을 가져온 다음 UI 상태를 원자적으로 전환합니다. 먼저 미리 가져오면 기록 로드에 실패할 때 활성 채팅이 지워지는 것을
        방지합니다.

Args:
            thread_id: 재개할 스레드 ID입니다.

        """
        if not self._agent:
            await self._mount_message(
                AppMessage("Cannot switch threads: no active agent")
            )
            return

        if not self._session_state:
            await self._mount_message(
                AppMessage("Cannot switch threads: no active session")
            )
            return

        # Skip if already on this thread
        if self._session_state.thread_id == thread_id:
            await self._mount_message(AppMessage(f"Already on thread: {thread_id}"))
            return

        if self._thread_switching:
            await self._mount_message(AppMessage("Thread switch already in progress."))
            return

        # Save previous state for rollback on failure
        prev_thread_id = self._lc_thread_id
        prev_session_thread = self._session_state.thread_id
        self._thread_switching = True
        if self._chat_input:
            self._chat_input.set_cursor_active(active=False)

        prefetched_payload: _ThreadHistoryPayload | None = None
        try:
            self._update_status(f"Loading thread: {thread_id}")
            prefetched_payload = await self._fetch_thread_history_data(thread_id)

            # Clear conversation (similar to /clear, without creating a new thread)
            self._pending_messages.clear()
            self._queued_widgets.clear()
            await self._clear_messages()
            self._context_tokens = 0
            self._tokens_approximate = False
            self._update_tokens(0)
            self._update_status("")

            # Switch to the selected thread
            self._session_state.thread_id = thread_id
            self._lc_thread_id = thread_id

            self._update_welcome_banner(
                thread_id,
                missing_message="Welcome banner not found during thread switch to %s",
                warn_if_missing=False,
            )

            # Load thread history
            await self._load_thread_history(
                thread_id=thread_id,
                preloaded_payload=prefetched_payload,
            )
        except Exception as exc:
            if prefetched_payload is None:
                logger.exception("Failed to prefetch history for thread %s", thread_id)
                await self._mount_message(
                    AppMessage(
                        f"Failed to switch to thread {thread_id}: {exc}. "
                        "Use /threads to try again."
                    )
                )
                return
            logger.exception("Failed to switch to thread %s", thread_id)
            # Restore previous thread IDs so the user can retry
            self._session_state.thread_id = prev_session_thread
            self._lc_thread_id = prev_thread_id
            self._update_welcome_banner(
                prev_session_thread,
                missing_message=(
                    "Welcome banner not found during rollback to thread %s; "
                    "banner may display stale thread ID"
                ),
                warn_if_missing=True,
            )
            rollback_restore_failed = False
            # Attempt to restore the previous thread's visible history
            try:
                await self._clear_messages()
                await self._load_thread_history(thread_id=prev_session_thread)
            except Exception:  # Resilient session state saving
                rollback_restore_failed = True
                msg = (
                    "Could not restore previous thread history after failed "
                    "switch to %s"
                )
                logger.warning(msg, thread_id, exc_info=True)
            error_message = f"Failed to switch to thread {thread_id}: {exc}."
            if rollback_restore_failed:
                error_message += " Previous thread history could not be restored."
            error_message += " Use /threads to try again."
            await self._mount_message(AppMessage(error_message))
        finally:
            self._thread_switching = False
            self._update_status("")
            if self._chat_input:
                self._chat_input.set_cursor_active(active=not self._agent_running)

    async def _switch_model(
        self,
        model_spec: str,
        *,
        extra_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """대화 기록을 보존하면서 새로운 모델로 전환하세요.

        이를 위해서는 서버 지원 대화형 세션이 필요합니다. `ConfigurableModelMiddleware`이 다음 호출 시 선택하는 모델 재정의를
        설정하므로 대화 스레드가 그대로 유지되고 서버를 다시 시작할 필요가 없습니다.

Args:
            model_spec: 전환할 모델 사양입니다.

                Can be in `provider: 모델 형식
                (e.g., `'anthropic: clude-sonnet-4-5'`) 또는 모델 이름만
                자동 감지를 위해.
            extra_kwargs: `--model-params`의 추가 생성자 kwargs.

        """
        from deepagents_cli.config import create_model, detect_provider, settings
        from deepagents_cli.model_config import (
            ModelSpec,
            get_credential_env_var,
            has_provider_credentials,
            save_recent_model,
        )

        logger.info("Switching model to %s", model_spec)

        if self._model_switching:
            await self._mount_message(AppMessage("Model switch already in progress."))
            return

        self._model_switching = True
        try:
            # Defensively strip leading colon in case of empty provider,
            # treat ":claude-opus-4-6" as "claude-opus-4-6"
            model_spec = model_spec.removeprefix(":")

            if not self._remote_agent():
                await self._mount_message(
                    ErrorMessage("Model switching requires a server-backed session.")
                )
                return

            parsed = ModelSpec.try_parse(model_spec)
            if parsed:
                provider: str | None = parsed.provider
                model_name = parsed.model
            else:
                model_name = model_spec
                provider = detect_provider(model_spec)

            # Check credentials
            has_creds = has_provider_credentials(provider) if provider else None
            if has_creds is False and provider is not None:
                env_var = get_credential_env_var(provider)
                detail = (
                    f"{env_var} is not set or is empty"
                    if env_var
                    else (
                        f"provider '{provider}' is not recognized. "
                        "Add it to ~/.deepagents/config.toml with an "
                        "api_key_env field"
                    )
                )
                await self._mount_message(
                    ErrorMessage(f"Missing credentials: {detail}")
                )
                return
            if has_creds is None and provider:
                logger.debug(
                    "Credentials for provider '%s' cannot be verified;"
                    " proceeding anyway",
                    provider,
                )

            # Check if already using this exact model
            if model_name == settings.model_name and (
                not provider or provider == settings.model_provider
            ):
                current = f"{settings.model_provider}:{settings.model_name}"
                await self._mount_message(AppMessage(f"Already using {current}"))
                return

            # Build the provider:model spec for the configurable middleware.
            display = model_spec
            if provider and not parsed:
                display = f"{provider}:{model_name}"

            try:
                create_model(
                    display,
                    extra_kwargs=extra_kwargs,
                    profile_overrides=self._profile_override,
                ).apply_to_settings()
            except Exception as exc:
                logger.exception("Failed to resolve model metadata for %s", display)
                await self._mount_message(
                    ErrorMessage(f"Failed to switch model: {exc}")
                )
                return

            # Set the model override for ConfigurableModelMiddleware.
            # The next stream call passes CLIContext via context= and the
            # middleware swaps the model per-invocation — no graph recreation.
            self._model_override = display
            self._model_params_override = extra_kwargs

            if self._status_bar:
                self._status_bar.set_model(
                    provider=settings.model_provider or "",
                    model=settings.model_name or "",
                )

            if not await asyncio.to_thread(save_recent_model, display):
                await self._mount_message(
                    ErrorMessage(
                        "Model switched for this session, but could not save "
                        "preference. Check permissions for ~/.deepagents/"
                    )
                )
            else:
                await self._mount_message(AppMessage(f"Switched to {display}"))
            logger.info("Model switched to %s (via configurable middleware)", display)

            # Anchor to bottom so the confirmation message is visible
            with suppress(NoMatches, ScreenStackError):
                self.query_one("#chat", VerticalScroll).anchor()
        finally:
            self._model_switching = False

    async def _set_default_model(self, model_spec: str) -> None:
        """현재 세션을 전환하지 않고 구성에서 기본 모델을 설정합니다.

        향후 CLI 실행 시 이 모델을 사용할 수 있도록 `~/.deepagents/config.toml`의 `[models].default`을
        업데이트합니다. 실행 중인 세션에는 영향을 주지 않습니다.

Args:
            model_spec: 모델 사양(예: `'anthropic:claude-opus-4-6'`)입니다.

        """
        from deepagents_cli.config import detect_provider
        from deepagents_cli.model_config import ModelSpec, save_default_model

        model_spec = model_spec.removeprefix(":")

        parsed = ModelSpec.try_parse(model_spec)
        if not parsed:
            provider = detect_provider(model_spec)
            if provider:
                model_spec = f"{provider}:{model_spec}"

        if await asyncio.to_thread(save_default_model, model_spec):
            await self._mount_message(AppMessage(f"Default model set to {model_spec}"))
        else:
            await self._mount_message(
                ErrorMessage(
                    "Could not save default model. Check permissions for ~/.deepagents/"
                )
            )

    async def _clear_default_model(self) -> None:
        """구성에서 기본 모델을 제거합니다.

        삭제 후 향후 실행은 `[models].recent` 또는 환경 자동 감지로 대체됩니다.

        """
        from deepagents_cli.model_config import clear_default_model

        if await asyncio.to_thread(clear_default_model):
            await self._mount_message(
                AppMessage(
                    "Default model cleared. "
                    "Future launches will use recent model or auto-detect."
                )
            )
        else:
            await self._mount_message(
                ErrorMessage(
                    "Could not clear default model. "
                    "Check permissions for ~/.deepagents/"
                )
            )


# ---------------------------------------------------------------------------
# App shutdown result and top-level runner
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AppResult:
    """텍스트 애플리케이션을 실행한 결과입니다."""

    return_code: int
    """종료 코드(성공의 경우 0, 오류의 경우 0이 아님)"""

    thread_id: str | None
    """종료 시 최종 스레드 ID입니다. 다음과 같은 경우 초기 스레드 ID와 다를 수 있습니다.
    사용자가 `/threads`을(를) 통해 스레드를 전환했습니다.
    """

    session_stats: SessionStats = field(default_factory=SessionStats)
    """세션의 모든 턴에 대한 누적 사용 통계입니다."""

    update_available: tuple[bool, str | None] = (False, None)
    """종료 후 업데이트 경고는 `(is_available, latest_version)`입니다."""


async def run_textual_app(
    *,
    agent: Any = None,  # noqa: ANN401
    assistant_id: str | None = None,
    backend: CompositeBackend | None = None,
    auto_approve: bool = False,
    cwd: str | Path | None = None,
    thread_id: str | None = None,
    resume_thread: str | None = None,
    initial_prompt: str | None = None,
    mcp_server_info: list[MCPServerInfo] | None = None,
    profile_override: dict[str, Any] | None = None,
    server_proc: ServerProcess | None = None,
    server_kwargs: dict[str, Any] | None = None,
    mcp_preload_kwargs: dict[str, Any] | None = None,
    model_kwargs: dict[str, Any] | None = None,
) -> AppResult:
    """텍스트 애플리케이션을 실행합니다.

    `server_kwargs`이 제공되면(그리고 `agent`은 `None`임) 앱은 "연결 중..." 배너와 함께 즉시 시작되고 백그라운드에서 서버를
    시작합니다.  서버 정리는 앱이 종료된 후 자동으로 처리됩니다.

Args:
        agent: 사전 구성된 LangGraph 에이전트(선택 사항).
        assistant_id: 메모리 저장을 위한 에이전트 식별자입니다.
        backend: 파일 작업을 위한 백엔드.
        auto_approve: 자동 승인을 활성화한 상태로 시작할지 여부입니다.
        cwd: 표시할 현재 작업 디렉터리입니다.
        thread_id: 세션의 스레드 ID입니다.

            `resume_thread`이 제공되면 `None`입니다(TUI는 최종 ID를 비동기식으로 확인합니다).
        resume_thread: `-r` 플래그의 원시 재개 의도입니다. `-r`의 경우 `'__MOST_RECENT__'`, `-r <id>`의
                       경우 스레드 ID 문자열, 새 세션의 경우 `None`입니다.

            TUI 시작 중에 비동기적으로 해결되었습니다.
        initial_prompt: 세션이 시작될 때 자동 제출하라는 선택적 프롬프트입니다.
        mcp_server_info: `/mcp` 뷰어의 MCP 서버 메타데이터입니다.
        profile_override: `--profile-override`의 추가 프로필 필드는 모델 선택 세부 정보, 오프로드 예산 표시,
                          `/offload`와 같은 주문형 `create_model()` 호출을 포함하여 나중에 프로필 인식 동작이
                          CLI 재정의와 일관되게 유지되도록 유지됩니다.
        server_proc: 대화형 세션을 위한 LangGraph 서버 프로세스입니다.
        server_kwargs: 지연된 `start_server_and_get_agent` 호출에 대한 Kwargs입니다.
        mcp_preload_kwargs: 동시 MCP 메타데이터 사전 로드를 위한 Kwargs.
        model_kwargs: 지연된 `create_model()` 호출에 대한 Kwargs입니다.

            제공된 경우 모델 생성은 첫 번째 페인트 후 백그라운드 작업자에서 실행되므로 스플래시 화면이 즉시 나타납니다.

Returns:
        반환 코드와 최종 스레드 ID가 포함된 `AppResult`입니다.

    """
    app = DeepAgentsApp(
        agent=agent,
        assistant_id=assistant_id,
        backend=backend,
        auto_approve=auto_approve,
        cwd=cwd,
        thread_id=thread_id,
        resume_thread=resume_thread,
        initial_prompt=initial_prompt,
        mcp_server_info=mcp_server_info,
        profile_override=profile_override,
        server_proc=server_proc,
        server_kwargs=server_kwargs,
        mcp_preload_kwargs=mcp_preload_kwargs,
        model_kwargs=model_kwargs,
    )
    try:
        await app.run_async()
    finally:
        # Guarantee server cleanup regardless of how the app exits.
        # Covers both the pre-started server_proc path and the deferred
        # server_kwargs path (where the background worker sets _server_proc).
        if app._server_proc is not None:
            app._server_proc.stop()

    return AppResult(
        return_code=app.return_code or 0,
        thread_id=app._lc_thread_id,
        session_stats=app._session_stats,
        update_available=app._update_available,
    )
