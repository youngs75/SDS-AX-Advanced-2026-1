"""텍스트 채팅 UI 내에서 사람 승인 요청을 렌더링합니다.

이 모듈의 위젯은 CLI에서 사용하는 Human-In-The-Loop 흐름에 대한 도구 미리보기, 안전 경고 및 승인 선택 사항을 제공합니다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from textual.binding import Binding, BindingType
from textual.containers import Container, Vertical, VerticalScroll
from textual.content import Content
from textual.message import Message
from textual.widgets import Static

if TYPE_CHECKING:
    import asyncio

    from textual import events
    from textual.app import ComposeResult

from deepagents_cli import theme
from deepagents_cli.config import (
    SHELL_TOOL_NAMES,
    get_glyphs,
    is_ascii_mode,
)
from deepagents_cli.unicode_security import (
    check_url_safety,
    detect_dangerous_unicode,
    format_warning_detail,
    iter_string_values,
    looks_like_url_key,
    render_with_unicode_markers,
    strip_dangerous_unicode,
    summarize_issues,
)
from deepagents_cli.widgets.tool_renderers import get_renderer

# Max length for truncated shell command display
_SHELL_COMMAND_TRUNCATE_LENGTH: int = 120
_WARNING_PREVIEW_LIMIT: int = 3
_WARNING_TEXT_TRUNCATE_LENGTH: int = 220


class ApprovalMenu(Container):
    """표준 텍스트 패턴을 사용하는 승인 메뉴입니다.

    주요 디자인 결정(mistral-vibe 참조에 따름): - compose()가 포함된 컨테이너 기본 클래스 - 키 처리를 위한
    BINDINGS(on_key 아님) - can_focus_children = 포커스 도난 방지를 위한 False - 옵션에 대한 간단한 정적 위젯 -
    표준 메시지 게시 - 렌더러 패턴을 통한 도구별 위젯

    """

    can_focus = True
    can_focus_children = False

    # CSS is in app.tcss - no DEFAULT_CSS needed

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("up", "move_up", "Up", show=False),
        Binding("k", "move_up", "Up", show=False),
        Binding("down", "move_down", "Down", show=False),
        Binding("j", "move_down", "Down", show=False),
        Binding("enter", "select", "Select", show=False),
        Binding("1", "select_approve", "Approve", show=False),
        Binding("y", "select_approve", "Approve", show=False),
        Binding("2", "select_auto", "Auto-approve", show=False),
        Binding("a", "select_auto", "Auto-approve", show=False),
        Binding("3", "select_reject", "Reject", show=False),
        Binding("n", "select_reject", "Reject", show=False),
        Binding("e", "toggle_expand", "Expand command", show=False),
    ]

    class Decided(Message):
        """사용자가 결정을 내릴 때 전송되는 메시지입니다."""

        def __init__(self, decision: dict[str, str]) -> None:
            """사용자의 결정으로 Decided 메시지를 초기화합니다.

            Args:
                decision: 결정 유형(예: '승인', '거부' 또는 'auto_approve_all')이 포함된 사전입니다.

            """
            super().__init__()
            self.decision = decision

    # Tools that don't need detailed info display (already shown in tool call)
    _MINIMAL_TOOLS: ClassVar[frozenset[str]] = SHELL_TOOL_NAMES

    def __init__(
        self,
        action_requests: list[dict[str, Any]] | dict[str, Any],
        _assistant_id: str | None = None,
        id: str | None = None,  # noqa: A002  # Textual widget constructor uses `id` parameter
        **kwargs: Any,
    ) -> None:
        """ApprovalMenu 위젯을 초기화합니다.

        Args:
            action_requests: 단일 작업 요청 사전 또는 승인이 필요한 작업 요청 사전 목록입니다. 각 사전에는 'name'(도구 이름)
                             및 'args'(도구 인수)가 포함되어야 합니다.
            _assistant_id: 선택적 보조자 ID(현재는 사용되지 않으며 향후 사용을 위해 예약됨).
            id: 선택적 위젯 ID입니다. 기본값은 '승인 메뉴'입니다.
            **kwargs: 컨테이너 기본 클래스에 전달되는 추가 키워드 인수입니다.

        """
        super().__init__(id=id or "approval-menu", classes="approval-menu", **kwargs)
        # Support both single request (legacy) and list of requests (batch)
        if isinstance(action_requests, dict):
            self._action_requests = [action_requests]
        else:
            self._action_requests = action_requests

        # For display purposes, get tool names
        self._tool_names = [r.get("name", "unknown") for r in self._action_requests]
        self._selected = 0
        self._future: asyncio.Future[dict[str, str]] | None = None
        self._option_widgets: list[Static] = []
        self._tool_info_container: Vertical | None = None
        # Minimal display if ALL tools are bash/shell
        self._is_minimal = all(name in self._MINIMAL_TOOLS for name in self._tool_names)
        # For expandable shell commands
        self._command_expanded = False
        self._command_widget: Static | None = None
        self._has_expandable_command = self._check_expandable_command()
        self._security_warnings = self._collect_security_warnings()

    def set_future(self, future: asyncio.Future[dict[str, str]]) -> None:
        """사용자가 결정하면 해결될 미래를 설정합니다."""
        self._future = future

    def _check_expandable_command(self) -> bool:
        """확장할 수 있는 쉘 명령이 있는지 확인하세요.

        Returns:
            단일 작업 요청이 확장 가능한 셸 명령인지 여부입니다.

        """
        if len(self._action_requests) != 1:
            return False
        req = self._action_requests[0]
        if req.get("name", "") not in SHELL_TOOL_NAMES:
            return False
        command = str(req.get("args", {}).get("command", ""))
        return len(command) > _SHELL_COMMAND_TRUNCATE_LENGTH

    def _get_command_display(self, *, expanded: bool) -> Content:
        """명령 표시 내용(잘림 또는 전체)을 가져옵니다.

        Args:
            expanded: 전체 명령을 표시할지 아니면 잘린 버전을 표시할지 여부입니다.

        Returns:
            명령 표시에 대한 스타일이 지정된 콘텐츠입니다.

        Raises:
            RuntimeError: 빈 action_requests로 호출되는 경우.

        """
        if not self._action_requests:
            msg = "_get_command_display called with empty action_requests"
            raise RuntimeError(msg)
        req = self._action_requests[0]
        command_raw = str(req.get("args", {}).get("command", ""))
        command = strip_dangerous_unicode(command_raw)
        issues = detect_dangerous_unicode(command_raw)

        if expanded or len(command) <= _SHELL_COMMAND_TRUNCATE_LENGTH:
            command_display = command
        else:
            command_display = (
                command[:_SHELL_COMMAND_TRUNCATE_LENGTH] + get_glyphs().ellipsis
            )

        if not expanded and len(command) > _SHELL_COMMAND_TRUNCATE_LENGTH:
            display = Content.from_markup(
                "[bold]$cmd[/bold] [dim](press 'e' to expand)[/dim]",
                cmd=command_display,
            )
        else:
            display = Content.from_markup("[bold]$cmd[/bold]", cmd=command_display)

        if not issues:
            return display

        raw_with_markers = render_with_unicode_markers(command_raw)
        if not expanded and len(raw_with_markers) > _WARNING_TEXT_TRUNCATE_LENGTH:
            raw_with_markers = (
                raw_with_markers[:_WARNING_TEXT_TRUNCATE_LENGTH] + get_glyphs().ellipsis
            )

        return Content.assemble(
            display,
            Content.from_markup(
                "\n[yellow]Warning:[/yellow] hidden chars detected ($summary)\n"
                "[dim]raw: $raw[/dim]",
                summary=summarize_issues(issues),
                raw=raw_with_markers,
            ),
        )

    def compose(self) -> ComposeResult:
        """정적 하위 항목으로 위젯을 구성합니다.

        레이아웃: 먼저 도구 정보(승인 대상), 그 다음 하단에 옵션이 표시됩니다. bash/shell의 경우 도구 호출에 이미 표시되어 있으므로 도구
        정보를 건너뜁니다.

        Yields:
            제목, 도구 정보, 옵션 및 도움말 텍스트에 대한 위젯입니다.

        """
        # Title - show count if multiple tools
        count = len(self._action_requests)
        if count == 1:
            title = Content.from_markup(
                ">>> $name Requires Approval <<<", name=self._tool_names[0]
            )
        else:
            title = Content(f">>> {count} Tool Calls Require Approval <<<")
        yield Static(title, classes="approval-title")

        if self._security_warnings:
            parts: list[Content] = [
                Content.from_markup(
                    "[yellow]Warning:[/yellow] Potentially deceptive text"
                ),
            ]
            parts.extend(
                Content.from_markup("\n[dim]- $w[/dim]", w=warning)
                for warning in self._security_warnings[:_WARNING_PREVIEW_LIMIT]
            )
            if len(self._security_warnings) > _WARNING_PREVIEW_LIMIT:
                remaining = len(self._security_warnings) - _WARNING_PREVIEW_LIMIT
                parts.append(Content.styled(f"\n- +{remaining} more warning(s)", "dim"))
            yield Static(
                Content.assemble(*parts),
                classes="approval-security-warning",
            )

        # For shell commands, show the command (expandable if long)
        if self._is_minimal and len(self._action_requests) == 1:
            self._command_widget = Static(
                self._get_command_display(expanded=self._command_expanded),
                classes="approval-command",
            )
            yield self._command_widget

        # Tool info - only for non-minimal tools (diffs, writes show actual content)
        if not self._is_minimal:
            with VerticalScroll(classes="tool-info-scroll"):
                self._tool_info_container = Vertical(classes="tool-info-container")
                yield self._tool_info_container

            # Separator between tool details and options
            glyphs = get_glyphs()
            yield Static(glyphs.box_horizontal * 40, classes="approval-separator")

        # Options container at bottom
        with Container(classes="approval-options-container"):
            # Options - create 3 Static widgets
            for i in range(3):  # noqa: B007  # Loop variable unused - iterating for count only
                widget = Static("", classes="approval-option")
                self._option_widgets.append(widget)
                yield widget

        # Help text at the very bottom
        glyphs = get_glyphs()
        help_text = (
            f"{glyphs.arrow_up}/{glyphs.arrow_down} navigate {glyphs.bullet} "
            f"Enter select {glyphs.bullet} y/a/n quick keys {glyphs.bullet} Esc reject"
        )
        if self._has_expandable_command:
            help_text += f" {glyphs.bullet} e expand"
        yield Static(help_text, classes="approval-help")

    async def on_mount(self) -> None:
        """마운트 및 업데이트 도구 정보에 집중하세요."""
        if is_ascii_mode():
            colors = theme.get_theme_colors(self)
            self.styles.border = ("ascii", colors.warning)

        if not self._is_minimal:
            await self._update_tool_info()
        self._update_options()
        self.focus()

    async def _update_tool_info(self) -> None:
        """모든 도구에 대해 도구별 승인 위젯을 탑재합니다."""
        if not self._tool_info_container:
            return

        # Clear existing content
        await self._tool_info_container.remove_children()

        # Mount info for each tool
        for i, action_request in enumerate(self._action_requests):
            tool_name = action_request.get("name", "unknown")
            tool_args = action_request.get("args", {})

            # Add tool header if multiple tools
            if len(self._action_requests) > 1:
                header = Static(
                    Content.from_markup(
                        "[bold]$num. $name[/bold]",
                        num=i + 1,
                        name=tool_name,
                    )
                )
                await self._tool_info_container.mount(header)

            # Show description if present
            description = action_request.get("description")
            if description:
                desc_widget = Static(
                    Content.from_markup("[dim]$desc[/dim]", desc=description),
                    classes="approval-description",
                )
                await self._tool_info_container.mount(desc_widget)

            # Get the appropriate renderer for this tool
            renderer = get_renderer(tool_name)
            widget_class, data = renderer.get_approval_widget(tool_args)
            approval_widget = widget_class(data)
            await self._tool_info_container.mount(approval_widget)

    def _update_options(self) -> None:
        """선택 항목에 따라 옵션 위젯을 업데이트합니다."""
        count = len(self._action_requests)
        if count == 1:
            options = [
                "1. Approve (y)",
                "2. Auto-approve for this thread (a)",
                "3. Reject (n)",
            ]
        else:
            options = [
                f"1. Approve all {count} (y)",
                "2. Auto-approve for this thread (a)",
                f"3. Reject all {count} (n)",
            ]

        for i, (text, widget) in enumerate(
            zip(options, self._option_widgets, strict=True)
        ):
            cursor = f"{get_glyphs().cursor} " if i == self._selected else "  "
            widget.update(f"{cursor}{text}")

            # Update classes
            widget.remove_class("approval-option-selected")
            if i == self._selected:
                widget.add_class("approval-option-selected")

    def action_move_up(self) -> None:
        """선택 항목을 위로 이동합니다."""
        self._selected = (self._selected - 1) % 3
        self._update_options()

    def action_move_down(self) -> None:
        """선택 항목을 아래로 이동합니다."""
        self._selected = (self._selected + 1) % 3
        self._update_options()

    def action_select(self) -> None:
        """현재 옵션을 선택하세요."""
        self._handle_selection(self._selected)

    def action_select_approve(self) -> None:
        """승인 옵션을 선택하세요."""
        self._selected = 0
        self._update_options()
        self._handle_selection(0)

    def action_select_auto(self) -> None:
        """자동 승인 옵션을 선택하세요."""
        self._selected = 1
        self._update_options()
        self._handle_selection(1)

    def action_select_reject(self) -> None:
        """거부 옵션을 선택하세요."""
        self._selected = 2
        self._update_options()
        self._handle_selection(2)

    def action_toggle_expand(self) -> None:
        """쉘 명령 확장을 전환합니다."""
        if not self._has_expandable_command or not self._command_widget:
            return
        self._command_expanded = not self._command_expanded
        self._command_widget.update(
            self._get_command_display(expanded=self._command_expanded)
        )

    def _handle_selection(self, option: int) -> None:
        """선택한 옵션을 처리합니다."""
        decision_map = {
            0: "approve",
            1: "auto_approve_all",
            2: "reject",
        }
        decision = {"type": decision_map[option]}

        # Resolve the future
        if self._future and not self._future.done():
            self._future.set_result(decision)

        # Post message
        self.post_message(self.Decided(decision))

    def _collect_security_warnings(self) -> list[str]:
        """의심스러운 유니코드 및 URL 값에 대한 경고 문자열을 수집합니다.

        작업 인수에 중첩된 모든 문자열 값을 재귀적으로 검사합니다.

        Returns:
            현재 작업 요청 일괄 처리에 대한 경고 문자열입니다.

        """
        warnings: list[str] = []
        for action_request in self._action_requests:
            tool_name = str(action_request.get("name", "unknown"))
            args = action_request.get("args", {})
            if not isinstance(args, dict):
                continue
            for arg_path, text in iter_string_values(args):
                issues = detect_dangerous_unicode(text)
                if issues:
                    warnings.append(
                        f"{tool_name}.{arg_path}: hidden Unicode "
                        f"({summarize_issues(issues)})"
                    )
                if looks_like_url_key(arg_path):
                    result = check_url_safety(text)
                    if result.safe:
                        continue
                    detail = format_warning_detail(result.warnings)
                    if result.decoded_domain:
                        detail = f"{detail}; decoded host: {result.decoded_domain}"
                    warnings.append(f"{tool_name}.{arg_path}: {detail}")
        return warnings

    def on_blur(self, event: events.Blur) -> None:  # noqa: ARG002  # Textual event handler signature
        """결정이 내려질 때까지 초점을 가두기 위해 흐림에 다시 초점을 맞춥니다."""
        self.call_after_refresh(self.focus)
