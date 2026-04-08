"""검색된 MCP 서버 및 도구를 검사하기 위한 읽기 전용 모달입니다.

뷰어는 사용자가 로드된 서버와 현재 CLI 세션에 사용할 수 있는 도구 이름 또는 설명을 이해하는 데 도움이 됩니다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from textual.binding import Binding, BindingType
from textual.containers import Vertical, VerticalScroll
from textual.content import Content
from textual.events import (
    Click,  # noqa: TC002 - needed at runtime for Textual event dispatch
)
from textual.screen import ModalScreen
from textual.widgets import Static

if TYPE_CHECKING:
    from textual.app import ComposeResult

    from deepagents_cli.mcp_tools import MCPServerInfo

from deepagents_cli import theme
from deepagents_cli.config import get_glyphs, is_ascii_mode


class MCPToolItem(Static):
    """MCP 뷰어에서 선택 가능한 도구 항목입니다."""

    def __init__(
        self,
        name: str,
        description: str,
        index: int,
        *,
        classes: str = "",
    ) -> None:
        """도구 항목을 초기화합니다.

        Args:
            name: 도구 이름.
            description: 전체 도구 설명.
            index: 목록에서 이 도구의 플랫 인덱스입니다.
            classes: CSS 클래스.

        """
        if description:
            label = Content.from_markup(
                "  $name [dim]$desc[/dim]", name=name, desc=description
            )
        else:
            label = Content.from_markup("  $name", name=name)
        super().__init__(label, classes=classes)
        self.tool_name = name
        self.tool_description = description
        self.index = index
        self._expanded = False

    def _format_collapsed(self, name: str, description: str) -> Content:
        """축소된(한 줄) 레이블을 작성합니다.

        설명이 위젯 너비를 초과할 경우 `(...)`으로 설명을 자릅니다.

        Args:
            name: 도구 이름.
            description: 도구 설명.

        Returns:
            스타일이 지정된 콘텐츠 라벨.

        """
        if not description:
            return Content.from_markup("  $name", name=name)
        prefix_len = 2 + len(name) + 1
        avail = self.size.width - prefix_len - 1 if self.size.width else 0
        ellipsis = " (...)"
        if avail > 0 and len(description) > avail:
            cut = max(0, avail - len(ellipsis))
            desc_text = description[:cut] + ellipsis
        else:
            desc_text = description
        return Content.from_markup(
            "  $name [dim]$desc[/dim]", name=name, desc=desc_text
        )

    @staticmethod
    def _format_expanded(name: str, description: str) -> Content:
        """확장된(여러 줄) 레이블을 만듭니다.

        Args:
            name: 도구 이름.
            description: 도구 설명.

        Returns:
            다음 줄에 전체 설명이 포함된 스타일이 지정된 콘텐츠 라벨입니다.

        """
        if description:
            return Content.from_markup(
                "  [bold]$name[/bold]\n    [dim]$desc[/dim]",
                name=name,
                desc=description,
            )
        return Content.from_markup("  [bold]$name[/bold]", name=name)

    def toggle_expand(self) -> None:
        """축소된 보기와 확장된 보기 사이를 전환합니다."""
        self._expanded = not self._expanded
        if self._expanded:
            label = self._format_expanded(self.tool_name, self.tool_description)
            self.styles.height = "auto"
        else:
            label = self._format_collapsed(self.tool_name, self.tool_description)
            self.styles.height = 1
        self.update(label)

    def on_mount(self) -> None:
        """너비가 알려지면 올바른 잘림으로 다시 렌더링합니다."""
        if not self._expanded:
            self.update(self._format_collapsed(self.tool_name, self.tool_description))

    def on_resize(self) -> None:
        """위젯 너비가 변경되면 다시 잘립니다."""
        if not self._expanded:
            self.update(self._format_collapsed(self.tool_name, self.tool_description))

    def on_click(self, event: Click) -> None:
        """클릭 처리 - 상위 화면을 통해 확장을 선택하고 전환합니다.

        Args:
            event: 클릭 이벤트입니다.

        """
        event.stop()
        screen = self.screen
        if isinstance(screen, MCPViewerScreen):
            screen._move_to(self.index)
            self.toggle_expand()


class MCPViewerScreen(ModalScreen[None]):
    """활성 MCP 서버 및 해당 도구에 대한 모달 뷰어입니다.

    전송 유형 및 도구 개수와 함께 이름별로 그룹화된 서버를 표시합니다. 화살표 키로 탐색하고, 도구 설명을 확장/축소하려면 Enter를, 닫으려면
    Esc를 누르세요.

    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("up", "move_up", "Up", show=False, priority=True),
        Binding("k", "move_up", "Up", show=False, priority=True),
        Binding("down", "move_down", "Down", show=False, priority=True),
        Binding("j", "move_down", "Down", show=False, priority=True),
        Binding("enter", "toggle_expand", "Expand", show=False, priority=True),
        Binding("pageup", "page_up", "Page up", show=False, priority=True),
        Binding("pagedown", "page_down", "Page down", show=False, priority=True),
        Binding("escape", "cancel", "Close", show=False, priority=True),
    ]

    CSS = """
    MCPViewerScreen {
        align: center middle;
    }

    MCPViewerScreen > Vertical {
        width: 80;
        max-width: 90%;
        height: 80%;
        background: $surface;
        border: solid $primary;
        padding: 1 2;
    }

    MCPViewerScreen .mcp-viewer-title {
        text-style: bold;
        color: $primary;
        text-align: center;
        margin-bottom: 1;
    }

    MCPViewerScreen .mcp-list {
        height: 1fr;
        min-height: 5;
        scrollbar-gutter: stable;
        background: $background;
    }

    MCPViewerScreen .mcp-server-header {
        color: $primary;
        margin-top: 1;
    }

    MCPViewerScreen .mcp-list > .mcp-server-header:first-child {
        margin-top: 0;
    }

    MCPViewerScreen .mcp-tool-item {
        height: 1;
        padding: 0 1;
    }

    MCPViewerScreen .mcp-tool-item:hover {
        background: $surface-lighten-1;
    }

    MCPViewerScreen .mcp-tool-selected {
        background: $primary;
        text-style: bold;
    }

    MCPViewerScreen .mcp-tool-selected:hover {
        background: $primary-lighten-1;
    }

    MCPViewerScreen .mcp-empty {
        color: $text-muted;
        text-style: italic;
        text-align: center;
        margin-top: 2;
    }

    MCPViewerScreen .mcp-viewer-help {
        height: 1;
        color: $text-muted;
        text-style: italic;
        margin-top: 1;
        text-align: center;
    }
    """

    def __init__(self, server_info: list[MCPServerInfo]) -> None:
        """MCP 뷰어 화면을 초기화합니다.

        Args:
            server_info: 표시할 MCP 서버 메타데이터 목록입니다.

        """
        super().__init__()
        self._server_info = server_info
        self._tool_widgets: list[MCPToolItem] = []
        self._selected_index = 0

    def compose(self) -> ComposeResult:
        """화면 레이아웃을 구성합니다.

        Yields:
            MCP 뷰어 UI용 위젯입니다.

        """
        glyphs = get_glyphs()
        total_servers = len(self._server_info)
        total_tools = sum(len(s.tools) for s in self._server_info)

        with Vertical():
            if total_servers:
                server_label = "server" if total_servers == 1 else "servers"
                tool_label = "tool" if total_tools == 1 else "tools"
                title = (
                    f"MCP Servers ({total_servers} {server_label},"
                    f" {total_tools} {tool_label})"
                )
            else:
                title = "MCP Servers"
            yield Static(title, classes="mcp-viewer-title")

            with VerticalScroll(classes="mcp-list"):
                if not self._server_info:
                    yield Static(
                        "No MCP servers configured.\n"
                        "Use `--mcp-config` to load servers.",
                        classes="mcp-empty",
                    )
                else:
                    flat_index = 0
                    for server in self._server_info:
                        tool_count = len(server.tools)
                        t_label = "tool" if tool_count == 1 else "tools"
                        yield Static(
                            Content.from_markup(
                                "[bold]$name[/bold]"
                                f" [dim]$transport {glyphs.bullet}"
                                f" {tool_count} {t_label}[/dim]",
                                name=server.name,
                                transport=server.transport,
                            ),
                            classes="mcp-server-header",
                        )
                        for tool in server.tools:
                            classes = "mcp-tool-item"
                            if flat_index == 0:
                                classes += " mcp-tool-selected"
                            widget = MCPToolItem(
                                name=tool.name,
                                description=tool.description,
                                index=flat_index,
                                classes=classes,
                            )
                            self._tool_widgets.append(widget)
                            yield widget
                            flat_index += 1

            help_text = (
                f"{glyphs.arrow_up}/{glyphs.arrow_down} navigate"
                f" {glyphs.bullet} Enter expand/collapse"
                f" {glyphs.bullet} Esc close"
            )
            yield Static(help_text, classes="mcp-viewer-help")

    async def on_mount(self) -> None:
        """필요한 경우 ASCII 테두리 대체를 적용합니다."""
        if is_ascii_mode():
            container = self.query_one(Vertical)
            colors = theme.get_theme_colors(self)
            container.styles.border = ("ascii", colors.success)

    def _move_to(self, index: int) -> None:
        """선택 항목을 지정된 인덱스로 이동합니다.

        Args:
            index: 대상 도구 색인.

        """
        if not self._tool_widgets:
            return
        old = self._selected_index
        self._selected_index = index

        if old != index:
            self._tool_widgets[old].remove_class("mcp-tool-selected")
            self._tool_widgets[index].add_class("mcp-tool-selected")
            self._tool_widgets[index].scroll_visible()

    def _move_selection(self, delta: int) -> None:
        """델타 위치별로 선택 항목을 이동합니다.

        Args:
            delta: 이동할 위치 수입니다.

        """
        if not self._tool_widgets:
            return
        count = len(self._tool_widgets)
        target = (self._selected_index + delta) % count
        self._move_to(target)

    def action_move_up(self) -> None:
        """선택 항목을 위로 이동합니다."""
        self._move_selection(-1)

    def action_move_down(self) -> None:
        """선택 항목을 아래로 이동합니다."""
        self._move_selection(1)

    def action_toggle_expand(self) -> None:
        """선택한 도구에서 확장/축소를 전환합니다."""
        if self._tool_widgets:
            self._tool_widgets[self._selected_index].toggle_expand()

    def action_page_up(self) -> None:
        """한 페이지 위로 스크롤합니다."""
        scroll = self.query_one(".mcp-list", VerticalScroll)
        scroll.scroll_page_up()

    def action_page_down(self) -> None:
        """한 페이지 아래로 스크롤합니다."""
        scroll = self.query_one(".mcp-list", VerticalScroll)
        scroll.scroll_page_down()

    def action_cancel(self) -> None:
        """뷰어를 닫습니다."""
        self.dismiss(None)
