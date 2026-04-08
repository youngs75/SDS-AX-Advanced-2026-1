"""실시간 미리보기 동작이 포함된 모달 테마 선택기입니다.

이 모듈을 통해 사용자는 등록된 테마를 찾아보고 즉시 미리 볼 수 있으며 시각적 변경 사항을 확인하거나 취소할 수 있습니다.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option

if TYPE_CHECKING:
    from textual.app import ComposeResult

from deepagents_cli import theme
from deepagents_cli.config import get_glyphs, is_ascii_mode

logger = logging.getLogger(__name__)


class ThemeSelectorScreen(ModalScreen[str | None]):
    """실시간 미리보기가 포함된 테마 선택을 위한 모달 대화상자입니다.

    `OptionList`에 사용 가능한 테마를 표시합니다. 옵션 목록을 탐색하면 앱 테마를 교체하여 실시간 미리보기가 적용됩니다. Enter 키를 누르면
    선택한 테마 이름을 반환하고 Esc 키를 누르면 `None`을 반환합니다(원래 테마 복원).

    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    CSS = """
    ThemeSelectorScreen {
        align: center middle;
        background: transparent;
    }

    ThemeSelectorScreen > Vertical {
        width: 50;
        max-width: 90%;
        height: auto;
        max-height: 80%;
        background: $surface;
        border: solid $primary;
        padding: 1 2;
    }

    ThemeSelectorScreen .theme-selector-title {
        text-style: bold;
        color: $primary;
        text-align: center;
        margin-bottom: 1;
    }

    ThemeSelectorScreen OptionList {
        height: auto;
        max-height: 16;
        background: $background;
    }

    ThemeSelectorScreen .theme-selector-help {
        height: 1;
        color: $text-muted;
        text-style: italic;
        margin-top: 1;
        text-align: center;
    }
    """

    def __init__(self, current_theme: str) -> None:
        """ThemeSelectorScreen을 초기화합니다.

        Args:
            current_theme: 현재 활성화된 테마 이름(강조표시용)입니다.

        """
        super().__init__()
        self._current_theme = current_theme
        self._original_theme = current_theme

    def compose(self) -> ComposeResult:
        """화면 레이아웃을 구성합니다.

        Yields:
            테마 선택기 UI용 위젯입니다.

        """
        glyphs = get_glyphs()
        options: list[Option] = []
        highlight_index = 0

        for i, (name, entry) in enumerate(theme.ThemeEntry.REGISTRY.items()):
            label = entry.label
            if name == self._current_theme:
                label = f"{label} (current)"
                highlight_index = i
            options.append(Option(label, id=name))

        with Vertical():
            yield Static("Select Theme", classes="theme-selector-title")
            option_list = OptionList(*options, id="theme-options")
            option_list.highlighted = highlight_index
            yield option_list
            help_text = (
                f"{glyphs.arrow_up}/{glyphs.arrow_down} preview"
                f" {glyphs.bullet} Enter select"
                f" {glyphs.bullet} Esc cancel"
            )
            yield Static(help_text, classes="theme-selector-help")

    def on_mount(self) -> None:
        """필요한 경우 ASCII 테두리를 적용합니다."""
        if is_ascii_mode():
            container = self.query_one(Vertical)
            colors = theme.get_theme_colors(self)
            container.styles.border = ("ascii", colors.success)

    def on_option_list_option_highlighted(
        self, event: OptionList.OptionHighlighted
    ) -> None:
        """강조 표시된 테마를 실시간으로 미리 봅니다.

        Args:
            event: 옵션이 강조된 이벤트입니다.

        """
        name = event.option.id
        if name is not None and name in theme.ThemeEntry.REGISTRY:
            try:
                self.app.theme = name
                # refresh_css only repaints the active (modal) screen's layout;
                # force the screen beneath us to repaint so the user sees the
                # preview through the transparent scrim.
                stack = self.app.screen_stack
                if len(stack) > 1:
                    stack[-2].refresh(layout=True)
            except Exception:
                logger.warning("Failed to preview theme '%s'", name, exc_info=True)
                try:
                    self.app.theme = self._original_theme
                except Exception:
                    logger.warning(
                        "Failed to restore original theme '%s'",
                        self._original_theme,
                        exc_info=True,
                    )

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """선택한 테마를 커밋합니다.

        Args:
            event: 옵션선택 이벤트입니다.

        """
        name = event.option.id
        if name is not None and name in theme.ThemeEntry.REGISTRY:
            self.dismiss(name)
        else:
            logger.warning("Selected theme '%s' is no longer available", name)
            self.dismiss(None)

    def action_cancel(self) -> None:
        """원래 테마를 복원하고 닫습니다."""
        self.app.theme = self._original_theme
        self.dismiss(None)
