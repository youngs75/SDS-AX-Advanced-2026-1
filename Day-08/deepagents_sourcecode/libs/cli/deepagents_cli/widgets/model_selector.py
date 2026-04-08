"""`/model` 명령에서 사용되는 모달 모델 선택기입니다.

화면에는 검색 가능한 공급자/모델 옵션, 현재 선택 사항, 저장된 기본값이나 활성 세션 모델을 업데이트하는 작업이 표시됩니다.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, ClassVar

from textual.binding import Binding, BindingType
from textual.containers import Container, Vertical, VerticalScroll
from textual.content import Content
from textual.events import (
    Click,  # noqa: TC002 - needed at runtime for Textual event dispatch
)
from textual.fuzzy import Matcher
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Input, Static

if TYPE_CHECKING:
    from collections.abc import Mapping

    from textual.app import ComposeResult

from deepagents_cli import theme
from deepagents_cli.config import Glyphs, get_glyphs, is_ascii_mode
from deepagents_cli.model_config import (
    ModelConfig,
    ModelProfileEntry,
    clear_default_model,
    get_available_models,
    get_model_profiles,
    has_provider_credentials,
    save_default_model,
)

logger = logging.getLogger(__name__)


class ModelOption(Static):
    """선택기에서 클릭 가능한 모델 옵션입니다."""

    def __init__(
        self,
        label: str | Content,
        model_spec: str,
        provider: str,
        index: int,
        *,
        has_creds: bool | None = True,
        classes: str = "",
    ) -> None:
        """모델 옵션을 초기화합니다.

        Args:
            label: 표시 콘텐츠 — `Content` 개체(선호) 또는 `Static`이 마크업으로 구문 분석할 일반 문자열입니다.
            model_spec: 모델 사양(공급자:모델 형식)입니다.
            provider: 공급자 이름입니다.
            index: 필터링된 목록에 있는 이 옵션의 인덱스입니다.
            has_creds: 공급자가 유효한 자격 증명을 가지고 있는지 여부. 확인된 경우 True, 누락된 경우 False, 알 수 없는 경우
                       None입니다.
            classes: 스타일링을 위한 CSS 클래스.

        """
        super().__init__(label, classes=classes)
        self.model_spec = model_spec
        self.provider = provider
        self.index = index
        self.has_creds = has_creds

    class Clicked(Message):
        """모델 옵션을 클릭하면 전송되는 메시지입니다."""

        def __init__(self, model_spec: str, provider: str, index: int) -> None:
            """Clicked 메시지를 초기화합니다.

            Args:
                model_spec: 모델 사양입니다.
                provider: 공급자 이름입니다.
                index: 클릭한 옵션의 인덱스입니다.

            """
            super().__init__()
            self.model_spec = model_spec
            self.provider = provider
            self.index = index

    def on_click(self, event: Click) -> None:
        """이 옵션을 클릭하세요.

        Args:
            event: 클릭 이벤트입니다.

        """
        event.stop()
        self.post_message(self.Clicked(self.model_spec, self.provider, self.index))


class ModelSelectorScreen(ModalScreen[tuple[str, str] | None]):
    """모델 선택을 위한 전체 화면 모달입니다.

    키보드 탐색 및 검색 필터링을 통해 제공업체별로 그룹화된 사용 가능한 모델을 표시합니다. 현재 모델이 강조 표시됩니다.

    선택 시 (model_spec, 제공자) 튜플을 반환하거나 취소 시 None을 반환합니다.

    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("up", "move_up", "Up", show=False, priority=True),
        Binding("k", "move_up", "Up", show=False, priority=True),
        Binding("down", "move_down", "Down", show=False, priority=True),
        Binding("j", "move_down", "Down", show=False, priority=True),
        Binding("tab", "tab_complete", "Tab complete", show=False, priority=True),
        Binding("pageup", "page_up", "Page up", show=False, priority=True),
        Binding("pagedown", "page_down", "Page down", show=False, priority=True),
        Binding("enter", "select", "Select", show=False, priority=True),
        Binding("ctrl+s", "set_default", "Set default", show=False, priority=True),
        Binding("escape", "cancel", "Cancel", show=False, priority=True),
    ]

    CSS = """
    ModelSelectorScreen {
        align: center middle;
    }

    ModelSelectorScreen > Vertical {
        width: 80;
        max-width: 90%;
        height: 80%;
        background: $surface;
        border: solid $primary;
        padding: 1 2;
    }

    ModelSelectorScreen .model-selector-title {
        text-style: bold;
        color: $primary;
        text-align: center;
        margin-bottom: 1;
    }

    ModelSelectorScreen #model-filter {
        margin-bottom: 1;
        border: solid $primary-lighten-2;
    }

    ModelSelectorScreen #model-filter:focus {
        border: solid $primary;
    }

    ModelSelectorScreen .model-list {
        height: 1fr;
        min-height: 5;
        scrollbar-gutter: stable;
        background: $background;
    }

    ModelSelectorScreen #model-options {
        height: auto;
    }

    ModelSelectorScreen .model-provider-header {
        color: $primary;
        margin-top: 1;
    }

    ModelSelectorScreen #model-options > .model-provider-header:first-child {
        margin-top: 0;
    }

    ModelSelectorScreen .model-option {
        height: 1;
        padding: 0 1;
    }

    ModelSelectorScreen .model-option:hover {
        background: $surface-lighten-1;
    }

    ModelSelectorScreen .model-option-selected {
        background: $primary;
        color: $background;
        text-style: bold;
    }

    ModelSelectorScreen .model-option-selected:hover {
        background: $primary-lighten-1;
    }

    ModelSelectorScreen .model-option-current {
        text-style: italic;
    }

    ModelSelectorScreen .model-selector-help {
        height: 1;
        color: $text-muted;
        text-style: italic;
        margin-top: 1;
        text-align: center;
    }

    ModelSelectorScreen .model-detail-footer {
        height: 4;
        padding: 0 2;
        margin-top: 1;
    }
    """

    def __init__(
        self,
        current_model: str | None = None,
        current_provider: str | None = None,
        cli_profile_override: dict[str, Any] | None = None,
    ) -> None:
        """ModelSelectorScreen을 초기화합니다.

        데이터 로드(모델 검색, 프로필)는 `on_mount`로 연기되므로 화면이 즉시 푸시되고 비동기식으로 채워집니다.

        Args:
            current_model: 현재 활성 모델 이름(강조 표시)
            current_provider: 현재 모델의 공급자입니다.
            cli_profile_override: `--profile-override`의 추가 프로필 필드입니다.

                CLI 재정의가 세부 바닥글에 `*` 마커와 함께 표시되도록 업스트림 + config.toml 프로필 위에 병합됩니다.

        """
        super().__init__()
        self._current_model = current_model
        self._current_provider = current_provider
        self._cli_profile_override = cli_profile_override

        # Model data — populated asynchronously in on_mount via _load_model_data
        self._all_models: list[tuple[str, str]] = []
        self._filtered_models: list[tuple[str, str]] = []
        self._selected_index = 0
        self._options_container: Container | None = None
        self._option_widgets: list[ModelOption] = []
        self._filter_text = ""
        self._current_spec: str | None = None
        if current_model and current_provider:
            self._current_spec = f"{current_provider}:{current_model}"
        self._default_spec: str | None = None
        self._profiles: Mapping[str, ModelProfileEntry] = {}
        self._loaded = False

    def _find_current_model_index(self) -> int:
        """필터링된 목록에서 현재 모델의 인덱스를 찾습니다.

        Returns:
            현재 모델의 인덱스이거나, 찾을 수 없으면 0입니다.

        """
        if not self._current_model or not self._current_provider:
            return 0

        current_spec = f"{self._current_provider}:{self._current_model}"
        for i, (model_spec, _) in enumerate(self._filtered_models):
            if model_spec == current_spec:
                return i
        return 0

    def compose(self) -> ComposeResult:
        """화면 레이아웃을 구성합니다.

        Yields:
            모델 선택기 UI용 위젯입니다.

        """
        glyphs = get_glyphs()

        with Vertical():
            # Title with current model in provider:model format
            if self._current_model and self._current_provider:
                current_spec = f"{self._current_provider}:{self._current_model}"
                title = f"Select Model (current: {current_spec})"
            elif self._current_model:
                title = f"Select Model (current: {self._current_model})"
            else:
                title = "Select Model"
            yield Static(title, classes="model-selector-title")

            # Search input
            yield Input(
                placeholder="Type to filter or enter provider:model...",
                id="model-filter",
            )

            # Scrollable model list
            with VerticalScroll(classes="model-list"):
                self._options_container = Container(id="model-options")
                yield self._options_container

            # Model detail footer
            yield Static("", classes="model-detail-footer", id="model-detail-footer")

            # Help text
            help_text = (
                f"{glyphs.arrow_up}/{glyphs.arrow_down} navigate"
                f" {glyphs.bullet} Enter select"
                f" {glyphs.bullet} Ctrl+S set default"
                f" {glyphs.bullet} Esc cancel"
            )
            yield Static(help_text, classes="model-selector-help")

    @staticmethod
    def _load_model_data(
        cli_override: dict[str, Any] | None,
    ) -> tuple[
        list[tuple[str, str]],
        str | None,
        Mapping[str, ModelProfileEntry],
    ]:
        """모델 검색 데이터를 동기식으로 수집합니다.

        `asyncio.to_thread`을 통해 호출되도록 의도되었으므로 `get_available_models`의 파일 시스템 I/O가 이벤트
        루프를 차단하지 않습니다.

        Returns:
            (all_models, default_spec, 프로필)의 튜플
                `all_models` is a list of `(provider: 모델 사양, 공급자)`
                쌍에서 `default_spec`은 구성된 기본 모델 또는 `None`이고 `profiles`는 사양 문자열을 프로필 항목에
                매핑합니다.

        """
        all_models: list[tuple[str, str]] = [
            (f"{provider}:{model}", provider)
            for provider, models in get_available_models().items()
            for model in models
        ]

        config = ModelConfig.load()
        profiles = get_model_profiles(cli_override=cli_override)
        return all_models, config.default_model, profiles

    async def on_mount(self) -> None:
        """마운트 시 화면을 설정합니다.

        화면 프레임이 즉시 렌더링되도록 배경 스레드에 모델 데이터를 로드한 다음 모델 목록을 채웁니다.

        """
        if is_ascii_mode():
            colors = theme.get_theme_colors(self)
            container = self.query_one(Vertical)
            container.styles.border = ("ascii", colors.success)

        # Focus the filter input immediately so the user can start typing
        # while model data loads.
        filter_input = self.query_one("#model-filter", Input)
        filter_input.focus()

        # Offload to thread because get_available_models does filesystem I/O
        try:
            all_models, default_spec, profiles = await asyncio.to_thread(
                self._load_model_data, self._cli_profile_override
            )
        except Exception:
            logger.exception("Failed to load model data for /model selector")
            self._loaded = True
            if self.is_running:
                self.notify(
                    "Could not load model list. "
                    "Check provider packages and config.toml.",
                    severity="error",
                    timeout=10,
                    markup=False,
                )
                await self._update_display()
                self._update_footer()
            return

        # Screen may have been dismissed while the thread was running
        if not self.is_running:
            return

        self._all_models = all_models
        self._default_spec = default_spec
        self._profiles = profiles
        self._filtered_models = list(self._all_models)
        self._selected_index = self._find_current_model_index()
        self._loaded = True

        # Re-apply any filter text the user typed while data was loading
        if self._filter_text:
            self._update_filtered_list()

        await self._update_display()
        self._update_footer()

    def on_input_changed(self, event: Input.Changed) -> None:
        """모델을 사용자 유형으로 필터링합니다.

        Args:
            event: 입력이 변경된 이벤트입니다.

        """
        self._filter_text = event.value
        if not self._loaded:
            return  # on_mount will re-apply filter after data loads
        self._update_filtered_list()
        self.call_after_refresh(self._update_display)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """필터 입력이 포커스되면 Enter 키를 처리합니다.

        Args:
            event: 입력이 제출된 이벤트입니다.

        """
        event.stop()
        self.action_select()

    def on_model_option_clicked(self, event: ModelOption.Clicked) -> None:
        """모델 옵션을 클릭하세요.

        Args:
            event: 모델 정보가 포함된 클릭 이벤트입니다.

        """
        self._selected_index = event.index
        self.dismiss((event.model_spec, event.provider))

    def _update_filtered_list(self) -> None:
        """퍼지 일치를 사용하여 검색 텍스트를 기반으로 필터링된 모델을 업데이트합니다.

        결과는 일치 점수에 따라 정렬됩니다(가장 좋은 것부터).

        """
        query = self._filter_text.strip()
        if not query:
            self._filtered_models = list(self._all_models)
            self._selected_index = self._find_current_model_index()
            return

        tokens = query.split()

        try:
            matchers = [Matcher(token, case_sensitive=False) for token in tokens]
            scored: list[tuple[float, str, str]] = []
            for spec, provider in self._all_models:
                scores = [m.match(spec) for m in matchers]
                if all(s > 0 for s in scores):
                    scored.append((min(scores), spec, provider))
        except Exception:
            # graceful fallback if Matcher fails on edge-case input
            logger.warning(
                "Fuzzy matcher failed for query %r, falling back to full list",
                query,
                exc_info=True,
            )
            self._filtered_models = list(self._all_models)
            self._selected_index = self._find_current_model_index()
            return

        self._filtered_models = [
            (spec, provider) for score, spec, provider in sorted(scored, reverse=True)
        ]
        self._selected_index = 0

    async def _update_display(self) -> None:
        """공급자별로 그룹화된 모델 목록을 렌더링합니다.

        전체 DOM 재구축을 수행합니다(모든 하위 요소 제거, 다시 마운트). 화살표 키 탐색에서는 전체 재구축 비용을 피하기 위해 대신
        `_move_selection`을 사용합니다.

        """
        if not self._options_container:
            return

        await self._options_container.remove_children()
        self._option_widgets = []

        if not self._filtered_models:
            msg = "Loading models…" if not self._loaded else "No matching models"
            await self._options_container.mount(Static(Content.styled(msg, "dim")))
            self._update_footer()
            return

        # Group by provider, preserving insertion order so models from the
        # same provider cluster together in the visual list.
        by_provider: dict[str, list[tuple[str, str]]] = {}
        for model_spec, provider in self._filtered_models:
            by_provider.setdefault(provider, []).append((model_spec, provider))

        # Rebuild _filtered_models to match the provider-grouped display
        # order. Without this, _filtered_models stays in score-sorted order
        # while _option_widgets follow provider-grouped order, causing
        # _update_footer to look up the wrong model for the highlighted
        # index.
        grouped_order: list[tuple[str, str]] = []
        for entries in by_provider.values():
            grouped_order.extend(entries)

        # Remap selected_index so the same model stays highlighted.
        old_spec = self._filtered_models[self._selected_index][0]
        self._filtered_models = grouped_order
        self._selected_index = next(
            (i for i, (s, _) in enumerate(grouped_order) if s == old_spec),
            0,
        )

        glyphs = get_glyphs()
        flat_index = 0
        selected_widget: ModelOption | None = None

        # Build current model spec for comparison
        current_spec = None
        if self._current_model and self._current_provider:
            current_spec = f"{self._current_provider}:{self._current_model}"

        # Resolve credentials upfront so the widget-building loop
        # stays focused on layout
        creds = {p: has_provider_credentials(p) for p in by_provider}

        # Collect all widgets first, then batch-mount once to avoid
        # individual DOM mutations per widget
        all_widgets: list[Static] = []

        for provider, model_entries in by_provider.items():
            # Provider header with credential indicator
            has_creds = creds[provider]
            if has_creds is True:
                cred_indicator = glyphs.checkmark
            elif has_creds is False:
                cred_indicator = f"{glyphs.warning} missing credentials"
            else:
                cred_indicator = f"{glyphs.question} credentials unknown"
            all_widgets.append(
                Static(
                    Content.from_markup(
                        "[bold]$provider[/bold] [dim]$cred[/dim]",
                        provider=provider,
                        cred=cred_indicator,
                    ),
                    classes="model-provider-header",
                )
            )

            for model_spec, _prov in model_entries:
                is_current = model_spec == current_spec
                is_selected = flat_index == self._selected_index

                classes = "model-option"
                if is_selected:
                    classes += " model-option-selected"
                if is_current:
                    classes += " model-option-current"

                label = self._format_option_label(
                    model_spec,
                    selected=is_selected,
                    current=is_current,
                    has_creds=has_creds,
                    is_default=model_spec == self._default_spec,
                    status=self._get_model_status(model_spec),
                )
                widget = ModelOption(
                    label=label,
                    model_spec=model_spec,
                    provider=provider,
                    index=flat_index,
                    has_creds=has_creds,
                    classes=classes,
                )
                all_widgets.append(widget)
                self._option_widgets.append(widget)

                if is_selected:
                    selected_widget = widget

                flat_index += 1

        await self._options_container.mount(*all_widgets)

        # Scroll the selected item into view without animation so the list
        # appears already scrolled to the current model on first paint.
        if selected_widget:
            if self._selected_index == 0:
                # First item: scroll to top so header is visible
                scroll_container = self.query_one(".model-list", VerticalScroll)
                scroll_container.scroll_home(animate=False)
            else:
                selected_widget.scroll_visible(animate=False)

        self._update_footer()

    @staticmethod
    def _format_option_label(
        model_spec: str,
        *,
        selected: bool,
        current: bool,
        has_creds: bool | None,
        is_default: bool = False,
        status: str | None = None,
    ) -> Content:
        """모델 옵션에 대한 표시 라벨을 작성합니다.

        Args:
            model_spec: `provider:model` 문자열.
            selected: 이 옵션이 현재 강조 표시되어 있는지 여부입니다.
            current: 활성 모델인지 여부입니다.
            has_creds: 자격 증명 상태(True/False/None).
            is_default: 구성된 기본 모델인지 여부입니다.
            status: 프로필의 모델 상태(예: `'deprecated'`, `'beta'`, `'alpha'`) `'deprecated'`은
                    빨간색으로 렌더링됩니다. None이 아닌 다른 값은 노란색으로 렌더링됩니다.

        Returns:
            스타일이 지정된 콘텐츠 라벨.

        """
        colors = theme.get_theme_colors()
        glyphs = get_glyphs()
        cursor = f"{glyphs.cursor} " if selected else "  "
        if not has_creds:
            spec = Content.styled(model_spec, colors.warning)
        elif is_default:
            spec = Content.styled(model_spec, colors.primary)
        else:
            spec = Content(model_spec)
        suffix = Content.styled(" (current)", "dim") if current else Content("")
        default_suffix = (
            Content.styled(" (default)", colors.primary) if is_default else Content("")
        )
        if status == "deprecated":
            status_suffix = Content.styled(" (deprecated)", colors.error)
        elif status:
            status_suffix = Content.styled(f" ({status})", colors.warning)
        else:
            status_suffix = Content("")
        return Content.assemble(cursor, spec, suffix, default_suffix, status_suffix)

    @staticmethod
    def _format_footer(
        profile_entry: ModelProfileEntry | None,
        glyphs: Glyphs,
    ) -> Content:
        """강조 표시된 모델에 대한 세부 바닥글 텍스트를 작성합니다.

        Args:
            profile_entry: 재정의 추적이 포함된 프로필 데이터 또는 없음.
            glyphs: 표시 문자에 대한 문자 집합입니다.

        Returns:
            4줄 바닥글의 경우 `Content` 스타일이 지정되었습니다.

        """
        from deepagents_cli.textual_adapter import format_token_count

        if profile_entry is None or not profile_entry["profile"]:
            return Content.styled("Model profile not available :(\n\n\n", "dim")

        profile = profile_entry["profile"]
        overridden = profile_entry["overridden_keys"]

        colors = theme.get_theme_colors()

        def _mark(key: str, text: str) -> Content:
            if key in overridden:
                return Content.styled(f"*{text}", colors.warning)
            return Content(text)

        def _format_token(key: str, suffix: str) -> Content | None:
            """토큰 수 프로필 키의 형식을 지정하고 원시 값으로 돌아갑니다.

            Returns:
                재정의 마커가 있는 `Content` 스타일 또는 키가 없는 경우 None입니다.

            """
            val = profile.get(key)
            if val is None:
                return None
            try:
                text = f"{format_token_count(int(val))} {suffix}"
            except (ValueError, TypeError, OverflowError):
                text = f"{val} {suffix}"
            return _mark(key, text)

        def _format_flags(keys: list[tuple[str, str]]) -> list[Content]:
            """부울 프로필 키를 녹색(켜짐) 또는 희미한(꺼짐) 레이블로 렌더링합니다.

            Returns:
                현재 키에 대한 스타일이 지정된 `Content` 개체 목록입니다.

            """
            parts: list[Content] = []
            for key, label in keys:
                if key in profile:
                    base = (
                        Content.styled(label, colors.success)
                        if profile[key]
                        else Content.styled(label, "dim")
                    )
                    if key in overridden:
                        base = Content.assemble(
                            Content.styled("*", colors.warning), base
                        )
                    parts.append(base)
            return parts

        # Line 1: Context window
        token_keys = [("max_input_tokens", "in"), ("max_output_tokens", "out")]
        ctx_parts = [p for k, s in token_keys if (p := _format_token(k, s)) is not None]
        bullet_sep = Content(f" {glyphs.bullet} ")
        line1 = (
            Content.assemble("Context: ", bullet_sep.join(ctx_parts))
            if ctx_parts
            else Content("")
        )

        # Line 2: Input modalities
        modality_keys = [
            ("text_inputs", "text"),
            ("image_inputs", "image"),
            ("audio_inputs", "audio"),
            ("pdf_inputs", "pdf"),
            ("video_inputs", "video"),
        ]
        modality_parts = _format_flags(modality_keys)
        space = Content(" ")
        line2 = (
            Content.assemble("Input: ", space.join(modality_parts))
            if modality_parts
            else Content("")
        )

        # Line 3: Capabilities
        capability_keys = [
            ("reasoning_output", "reasoning"),
            ("tool_calling", "tool calling"),
            ("structured_output", "structured output"),
        ]
        cap_parts = _format_flags(capability_keys)
        line3 = (
            Content.assemble("Capabilities: ", space.join(cap_parts))
            if cap_parts
            else Content("")
        )

        # Line 4: Override notice
        displayed_keys = {k for k, _ in token_keys + modality_keys + capability_keys}
        has_visible_override = bool(overridden & displayed_keys)
        line4 = (
            Content.from_markup("[dim][yellow]*[/yellow] = override[/dim]")
            if has_visible_override
            else Content("")
        )

        return Content.assemble(line1, "\n", line2, "\n", line3, "\n", line4)

    def _get_model_status(self, model_spec: str) -> str | None:
        """프로필에서 모델의 상태 필드를 찾아보세요.

        Args:
            model_spec: `provider:model` 문자열.

        Returns:
            모델에 `status` 키가 있는 프로필이 있는 경우 상태 문자열(예: `'deprecated'`), 그렇지 않으면 없음입니다.

        """
        entry = self._profiles.get(model_spec)
        if entry is None:
            return None
        profile = entry.get("profile")
        if not profile:
            return None
        return profile.get("status")

    def _update_footer(self) -> None:
        """현재 강조 표시된 모델의 세부 바닥글을 업데이트합니다."""
        footer = self.query_one("#model-detail-footer", Static)
        if not self._filtered_models:
            footer.update(Content.styled("No model selected", "dim"))
            return
        index = min(self._selected_index, len(self._filtered_models) - 1)
        spec, _ = self._filtered_models[index]
        entry = self._profiles.get(spec)
        try:
            text = self._format_footer(entry, get_glyphs())
        except (KeyError, ValueError, TypeError):  # Resilient footer rendering
            logger.warning("Failed to format footer for %s", spec, exc_info=True)
            text = Content.styled("Could not load profile details\n\n\n", "dim")
        footer.update(text)

    def _move_selection(self, delta: int) -> None:
        """선택 항목을 델타별로 이동하여 영향을 받은 위젯만 업데이트합니다.

        Args:
            delta: 이동할 위치 수(위로 -1, 아래로 +1)

        """
        if not self._filtered_models or not self._option_widgets:
            return

        count = len(self._filtered_models)
        old_index = self._selected_index
        new_index = (old_index + delta) % count
        self._selected_index = new_index

        # Update the previously selected widget
        old_widget = self._option_widgets[old_index]
        old_widget.remove_class("model-option-selected")
        old_widget.update(
            self._format_option_label(
                old_widget.model_spec,
                selected=False,
                current=old_widget.model_spec == self._current_spec,
                has_creds=old_widget.has_creds,
                is_default=old_widget.model_spec == self._default_spec,
                status=self._get_model_status(old_widget.model_spec),
            )
        )

        # Update the newly selected widget
        new_widget = self._option_widgets[new_index]
        new_widget.add_class("model-option-selected")
        new_widget.update(
            self._format_option_label(
                new_widget.model_spec,
                selected=True,
                current=new_widget.model_spec == self._current_spec,
                has_creds=new_widget.has_creds,
                is_default=new_widget.model_spec == self._default_spec,
                status=self._get_model_status(new_widget.model_spec),
            )
        )

        # Scroll the selected item into view
        if new_index == 0:
            scroll_container = self.query_one(".model-list", VerticalScroll)
            scroll_container.scroll_home(animate=False)
        else:
            new_widget.scroll_visible()

        self._update_footer()

    def action_move_up(self) -> None:
        """선택 항목을 위로 이동합니다."""
        self._move_selection(-1)

    def action_move_down(self) -> None:
        """선택 항목을 아래로 이동합니다."""
        self._move_selection(1)

    def action_tab_complete(self) -> None:
        """검색 텍스트를 현재 선택한 모델 사양으로 바꿉니다."""
        if not self._filtered_models:
            return
        model_spec, _ = self._filtered_models[self._selected_index]
        filter_input = self.query_one("#model-filter", Input)
        filter_input.value = model_spec
        filter_input.cursor_position = len(model_spec)

    def _visible_page_size(self) -> int:
        """하나의 시각적 페이지에 맞는 모델 옵션 수를 반환합니다.

        Returns:
            페이지당 모델 옵션 수는 1개 이상입니다.

        """
        default_page_size = 10
        try:
            scroll = self.query_one(".model-list", VerticalScroll)
            height = scroll.size.height
        except Exception:  # noqa: BLE001  # Fallback to default page size on any widget query error
            return default_page_size
        if height <= 0:
            return default_page_size

        total_models = len(self._filtered_models)
        if total_models == 0:
            return default_page_size

        # Each provider header = 1 row + margin-top: 1 (first has margin 0)
        num_headers = len(self.query(".model-provider-header"))
        header_rows = max(0, num_headers * 2 - 1) if num_headers else 0
        total_rows = total_models + header_rows
        return max(1, int(height * total_models / total_rows))

    def action_page_up(self) -> None:
        """선택 항목을 표시되는 한 페이지 위로 이동합니다."""
        if not self._filtered_models:
            return
        page = self._visible_page_size()
        target = max(0, self._selected_index - page)
        delta = target - self._selected_index
        if delta != 0:
            self._move_selection(delta)

    def action_page_down(self) -> None:
        """선택 항목을 표시되는 한 페이지 아래로 이동합니다."""
        if not self._filtered_models:
            return
        count = len(self._filtered_models)
        page = self._visible_page_size()
        target = min(count - 1, self._selected_index + page)
        delta = target - self._selected_index
        if delta != 0:
            self._move_selection(delta)

    def action_select(self) -> None:
        """현재 모델을 선택하세요."""
        # If there are filtered results, always select the highlighted model
        if self._filtered_models:
            model_spec, provider = self._filtered_models[self._selected_index]
            self.dismiss((model_spec, provider))
            return

        # No matches - check if user typed a custom provider:model spec
        filter_input = self.query_one("#model-filter", Input)
        custom_input = filter_input.value.strip()

        if custom_input and ":" in custom_input:
            provider = custom_input.split(":", 1)[0]
            self.dismiss((custom_input, provider))
        elif custom_input:
            self.dismiss((custom_input, ""))

    async def action_set_default(self) -> None:
        """강조 표시된 모델을 기본값으로 전환합니다.

        강조 표시된 모델이 이미 기본값인 경우 해당 모델을 지웁니다. 그렇지 않으면 새 기본값으로 설정됩니다.

        """
        if not self._filtered_models or not self._option_widgets:
            return

        model_spec, _provider = self._filtered_models[self._selected_index]
        help_widget = self.query_one(".model-selector-help", Static)

        if model_spec == self._default_spec:
            # Already default — clear it
            if await asyncio.to_thread(clear_default_model):
                self._default_spec = None
                self.call_after_refresh(self._update_display)
                help_widget.update(Content.styled("Default cleared", "bold"))
                self.set_timer(3.0, self._restore_help_text)
            else:
                help_widget.update(
                    Content.styled(
                        "Failed to clear default",
                        f"bold {theme.get_theme_colors(self).error}",
                    )
                )
                self.set_timer(3.0, self._restore_help_text)
        elif await asyncio.to_thread(save_default_model, model_spec):
            self._default_spec = model_spec
            self.call_after_refresh(self._update_display)
            help_widget.update(
                Content.from_markup(
                    "[bold]Default set to $spec[/bold]", spec=model_spec
                )
            )
            self.set_timer(3.0, self._restore_help_text)
        else:
            help_widget.update(
                Content.styled(
                    "Failed to save default",
                    f"bold {theme.get_theme_colors(self).error}",
                )
            )
            self.set_timer(3.0, self._restore_help_text)

    def _restore_help_text(self) -> None:
        """임시 메시지 뒤에 기본 도움말 텍스트를 복원합니다."""
        glyphs = get_glyphs()
        help_text = (
            f"{glyphs.arrow_up}/{glyphs.arrow_down} navigate"
            f" {glyphs.bullet} Enter select"
            f" {glyphs.bullet} Ctrl+S set default"
            f" {glyphs.bullet} Esc cancel"
        )
        help_widget = self.query_one(".model-selector-help", Static)
        help_widget.update(help_text)

    def action_cancel(self) -> None:
        """선택을 취소합니다."""
        self.dismiss(None)
