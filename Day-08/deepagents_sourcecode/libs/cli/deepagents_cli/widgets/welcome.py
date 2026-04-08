"""첫 번째 프롬프트 앞에 표시되는 초기 배너 및 바닥글 상태입니다.

이 모듈은 시작 시 연결, 실패 또는 유휴 중에 사용되는 브랜드 시작 보기, 회전 팁 및 다양한 바닥글 상태를 렌더링합니다.
"""

from __future__ import annotations

import asyncio
import random
from typing import TYPE_CHECKING, Any

from textual.color import Color as TColor
from textual.content import Content
from textual.style import Style as TStyle
from textual.widgets import Static

if TYPE_CHECKING:
    from textual.events import Click

from deepagents_cli import theme
from deepagents_cli._version import __version__
from deepagents_cli.config import (
    _get_editable_install_path,
    _is_editable_install,
    fetch_langsmith_project_url,
    get_banner,
    get_glyphs,
    get_langsmith_project_name,
)
from deepagents_cli.widgets._links import open_style_link

_TIPS: list[str] = [
    "Use @ to reference files and / for commands",
    "Try /threads to resume a previous conversation",
    "Use /offload when your conversation gets long",
    "Use /mcp to see your loaded tools and servers",
    "Use /remember to save learnings from this conversation",
    "Use /model to switch models mid-conversation",
    "Press ctrl+x to compose prompts in your external editor",
    "Press ctrl+u to delete to the start of the line in the chat input",
    "Use /skill:<name> to invoke a skill directly",
    "Type /update to check for and install updates",
    "Use /theme to customize the CLI colors and style",
    "Use /skill-creator to build reusable agent skills",
    "Use /auto-update to toggle automatic CLI updates",
]
"""환영 바닥글에 회전 팁이 표시됩니다.

세션당 한 명씩 선택됩니다.
"""


class WelcomeBanner(Static):
    """시작 시 표시되는 환영 배너입니다."""

    # Disable Textual's auto_links to prevent a flicker cycle: Style.__add__
    # calls .copy() for linked styles, generating a fresh random _link_id on
    # each render. This means highlight_link_id never stabilizes, causing an
    # infinite hover-refresh loop.
    auto_links = False

    DEFAULT_CSS = """
    WelcomeBanner {
        height: auto;
        padding: 1;
        margin-bottom: 1;
    }
    """

    def __init__(
        self,
        thread_id: str | None = None,
        mcp_tool_count: int = 0,
        *,
        connecting: bool = False,
        resuming: bool = False,
        local_server: bool = False,
        **kwargs: Any,
    ) -> None:
        """환영 배너를 초기화합니다.

        Args:
            thread_id: 배너에 표시할 선택적 스레드 ID입니다.
            mcp_tool_count: 시작 시 로드된 MCP 도구 수입니다.
            connecting: `True`인 경우 일반적인 준비 프롬프트 대신 "연결 중..." 바닥글을 표시합니다. 전환하려면
                        `set_connected`에 전화하세요.
            resuming: `True`인 경우 연결 바닥글에 `'Connecting...'` 변형 대신 "재개 중..."이라고 표시됩니다.
            local_server: `True`인 경우 연결 바닥글은 서버를 "로컬"(즉, CLI에서 관리하는 서버 프로세스)로 규정합니다.

                `resuming`이(가) `True`이면 무시됩니다.
            **kwargs: 추가 인수가 부모에게 전달되었습니다.

        """
        # Avoid collision with Widget._thread_id (Textual internal int)
        self._cli_thread_id: str | None = thread_id
        self._mcp_tool_count = mcp_tool_count
        self._connecting = connecting
        self._resuming = resuming
        self._local_server = local_server
        self._failed = False
        self._failure_error: str = ""
        self._project_name: str | None = get_langsmith_project_name()
        self._project_url: str | None = None
        self._tip: str = random.choice(_TIPS)  # noqa: S311

        super().__init__(self._build_banner(), **kwargs)

    def on_mount(self) -> None:
        """LangSmith 프로젝트 URL에 대한 백그라운드 가져오기를 시작합니다."""
        self.watch(self.app, "theme", self._on_theme_change, init=False)
        if self._project_name:
            self.run_worker(self._fetch_and_update, exclusive=True)

    def _on_theme_change(self) -> None:
        """앱 테마가 변경되면 배너를 다시 렌더링합니다."""
        self.update(self._build_banner(self._project_url))

    async def _fetch_and_update(self) -> None:
        """스레드에서 LangSmith URL을 가져오고 배너를 업데이트합니다."""
        if not self._project_name:
            return
        try:
            project_url = await asyncio.wait_for(
                asyncio.to_thread(fetch_langsmith_project_url, self._project_name),
                timeout=2.0,
            )
        except (TimeoutError, OSError):
            project_url = None
        if project_url:
            self._project_url = project_url
            self.update(self._build_banner(project_url))

    def update_thread_id(self, thread_id: str) -> None:
        """표시된 스레드 ID를 업데이트하고 배너를 다시 렌더링합니다.

        Args:
            thread_id: 표시할 새 스레드 ID입니다.

        """
        self._cli_thread_id = thread_id
        self.update(self._build_banner(self._project_url))

    def set_connected(self, mcp_tool_count: int = 0) -> None:
        """"연결 중"에서 "준비" 상태로 전환됩니다.

        Args:
            mcp_tool_count: 연결 중에 로드된 MCP 도구 수입니다.

        """
        self._connecting = False
        self._failed = False
        self._mcp_tool_count = mcp_tool_count
        self.update(self._build_banner(self._project_url))

    def set_failed(self, error: str) -> None:
        """"연결 중"에서 지속적인 실패 상태로 전환됩니다.

        Args:
            error: 서버 시작 실패를 설명하는 오류 메시지입니다.

        """
        self._connecting = False
        self._failed = True
        self._failure_error = error
        self.update(self._build_banner(self._project_url))

    def on_click(self, event: Click) -> None:  # noqa: PLR6301  # Textual event handler
        """한 번의 클릭으로 스타일이 포함된 하이퍼링크를 엽니다."""
        open_style_link(event)

    def _build_banner(self, project_url: str | None = None) -> Content:
        """배너 콘텐츠를 구축합니다.

        `project_url`이 제공되고 스레드 ID가 설정되면 스레드 ID는 LangSmith 스레드 보기에 대한 클릭 가능한 하이퍼링크로
        렌더링됩니다.

        Args:
            project_url: 프로젝트 이름과 스레드 ID를 연결하는 데 사용되는 LangSmith 프로젝트 URL입니다. `None`이면 링크
                         없이 텍스트가 렌더링됩니다.

        Returns:
            서식이 지정된 배너가 포함된 콘텐츠 개체입니다.

        """
        parts: list[str | tuple[str, str | TStyle] | Content] = []
        colors = theme.get_theme_colors(self)
        ansi = self.app.theme == "textual-ansi"

        banner = get_banner()
        primary_style: str | TStyle = (
            "bold"
            if ansi
            else TStyle(foreground=TColor.parse(colors.primary), bold=True)
        )

        if not ansi and _is_editable_install():
            # Highlight local-install version tag with tool accent; art stays primary.
            dev_style = TStyle(foreground=TColor.parse(colors.tool), bold=True)
            version_tag = f"v{__version__} (local)"
            idx = banner.rfind(version_tag)
            if idx >= 0:
                parts.extend(
                    [
                        (banner[:idx], primary_style),
                        (version_tag, dev_style),
                        (banner[idx + len(version_tag) :] + "\n", primary_style),
                    ]
                )
            else:
                parts.append((banner + "\n", primary_style))
        else:
            parts.append((banner + "\n", primary_style))

        # For ANSI theme, use "bold" (terminal foreground) instead of hex
        accent: str | TStyle = "bold" if ansi else colors.primary
        success_color: str = "bold green" if ansi else colors.success

        editable_path = _get_editable_install_path()
        if editable_path:
            parts.extend([("Installed from: ", "dim"), (editable_path, "dim"), "\n"])

        if self._project_name:
            parts.extend(
                [
                    (f"{get_glyphs().checkmark} ", success_color),
                    "LangSmith tracing: ",
                ]
            )
            if project_url:
                link_style: str | TStyle
                if ansi:
                    url = f"{project_url}?utm_source=deepagents-cli"
                    link_style = TStyle(bold=True, link=url)
                else:
                    link_style = TStyle(
                        foreground=TColor.parse(colors.primary),
                        link=f"{project_url}?utm_source=deepagents-cli",
                    )
                parts.append((f"'{self._project_name}'", link_style))
            else:
                parts.append((f"'{self._project_name}'", accent))
            parts.append("\n")

        if self._cli_thread_id:
            if project_url:
                thread_url = (
                    f"{project_url.rstrip('/')}/t/{self._cli_thread_id}"
                    "?utm_source=deepagents-cli"
                )
                parts.extend(
                    [
                        ("Thread: ", "dim"),
                        (self._cli_thread_id, TStyle(dim=True, link=thread_url)),
                        ("\n", "dim"),
                    ]
                )
            else:
                parts.append((f"Thread: {self._cli_thread_id}\n", "dim"))

        if self._mcp_tool_count > 0:
            parts.append((f"{get_glyphs().checkmark} ", success_color))
            label = "MCP tool" if self._mcp_tool_count == 1 else "MCP tools"
            parts.append(f"Loaded {self._mcp_tool_count} {label}\n")

        if self._failed:
            parts.append(build_failure_footer(self._failure_error))
        elif self._connecting:
            parts.append(
                build_connecting_footer(
                    resuming=self._resuming,
                    local_server=self._local_server,
                )
            )
        else:
            ready_color = "bold" if ansi else colors.primary
            parts.append(build_welcome_footer(primary_color=ready_color, tip=self._tip))
        return Content.assemble(*parts)


def build_failure_footer(error: str) -> Content:
    """서버 시작에 실패했을 때 표시되는 바닥글을 작성합니다.

    Args:
        error: 실패를 설명하는 오류 메시지입니다.

    Returns:
        지속적인 실패 메시지가 포함된 콘텐츠입니다.

    """
    colors = theme.get_theme_colors()
    return Content.assemble(
        ("\nServer failed to start: ", f"bold {colors.error}"),
        (error, colors.error),
        ("\n", colors.error),
    )


def build_connecting_footer(
    *, resuming: bool = False, local_server: bool = False
) -> Content:
    """서버 연결을 기다리는 동안 표시되는 바닥글을 작성합니다.

    Args:
        resuming: `'Connecting...'` 변형 대신 `'Resuming...'`을 표시합니다.
        local_server: 연결 메시지에서 서버를 "로컬"로 한정합니다.

            `resuming`이(가) `True`이면 무시됩니다.

    Returns:
        연결 상태 메시지가 포함된 콘텐츠입니다.

    """
    if resuming:
        text = "\nResuming...\n"
    elif local_server:
        text = "\nConnecting to local server...\n"
    else:
        text = "\nConnecting to server...\n"
    return Content.styled(text, "dim")


def build_welcome_footer(
    *, primary_color: str = theme.PRIMARY, tip: str | None = None
) -> Content:
    """환영 배너 하단에 표시된 바닥글을 작성합니다.

    사용자가 기능을 찾는 데 도움이 되는 팁이 포함되어 있습니다.

    Args:
        primary_color: 준비 프롬프트의 색상 문자열입니다.

            기본값은 모듈 수준 ANSI `PRIMARY` 상수입니다. 위젯 호출자는 활성 테마의 16진수 값을 전달해야 합니다.
        tip: 표시할 팁 텍스트입니다. `None`일 때 무작위 팁이 선택됩니다.

            다시 렌더링할 때 팁을 안정적으로 유지하려면 명시적인 값을 전달하세요.

    Returns:
        준비 메시지와 팁으로 만족하세요.

    """
    if tip is None:
        tip = random.choice(_TIPS)  # noqa: S311
    return Content.assemble(
        ("\nReady to code! What would you like to build?\n", primary_color),
        (f"Tip: {tip}", "dim italic"),
    )
