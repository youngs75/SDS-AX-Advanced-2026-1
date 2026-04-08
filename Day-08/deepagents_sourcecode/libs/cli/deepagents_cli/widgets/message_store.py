"""가상화된 채팅 기록을 위한 메시지 저장소입니다.

이 모듈은 메시지 가상화를 위한 데이터 구조 및 관리를 제공하므로 모든 메시지 데이터를 경량 데이터 클래스로 저장하는 동시에 DOM에 위젯의 슬라이딩 창만
유지하여 CLI가 대용량 메시지 기록을 효율적으로 처리할 수 있습니다.

이 접근 방식은 DOM에 `N` 줄만 유지하고 필요에 따라 이전 줄을 다시 만드는 Textual의 `Log` 위젯에서 영감을 받았습니다.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from time import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from textual.widget import Widget

logger = logging.getLogger(__name__)

# Fields on MessageData that callers are allowed to update via update_message().
# Prevents accidental overwriting of identity fields like id/type/timestamp.
_UPDATABLE_FIELDS: frozenset[str] = frozenset(
    {
        "content",
        "tool_status",
        "tool_output",
        "tool_expanded",
        "skill_expanded",
        "is_streaming",
        "height_hint",
    }
)


class MessageType(StrEnum):
    """채팅의 메시지 유형입니다."""

    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    SKILL = "skill"
    ERROR = "error"
    APP = "app"
    SUMMARIZATION = "summarization"
    DIFF = "diff"


class ToolStatus(StrEnum):
    """도구 호출 상태입니다."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"
    REJECTED = "rejected"
    SKIPPED = "skipped"


@dataclass
class MessageData:
    """가상화를 위한 메모리 내 메시지 데이터입니다.

    이 데이터 클래스에는 메시지 위젯을 다시 만드는 데 필요한 모든 정보가 포함되어 있습니다. 의미 있는 메모리 오버헤드 없이 수천 개의 메시지를 저장할
    수 있도록 경량으로 설계되었습니다.

    """

    type: MessageType
    """메시지 종류(사용자, 보조자, 도구 등)."""

    content: str
    """메시지의 기본 텍스트 콘텐츠입니다.

    대부분의 메시지 유형에서 이는 표시 텍스트입니다. TOOL 메시지의 경우 도구의 ID가 `tool_name` / `tool_args`에서 오기 때문에
    일반적으로 비어 있습니다.

    """

    id: str = field(default_factory=lambda: f"msg-{uuid.uuid4().hex[:8]}")
    """데이터 클래스를 해당 DOM 위젯과 일치시키는 데 사용되는 고유 식별자입니다."""

    timestamp: float = field(default_factory=time)
    """메시지가 생성된 시점의 Unix epoch 타임스탬프입니다."""

    # TOOL message fields - only populated for TOOL messages
    tool_name: str | None = None
    """호출된 도구의 이름입니다."""

    tool_args: dict[str, Any] | None = None
    """도구 호출에 인수가 전달되었습니다."""

    tool_status: ToolStatus | None = None
    """도구 호출의 현재 실행 상태입니다."""

    tool_output: str | None = None
    """실행 후 도구에서 반환된 출력입니다."""

    tool_expanded: bool = False
    """도구 출력 섹션이 UI에서 확장되는지 여부입니다."""

    # ---

    diff_file_path: str | None = None
    """diff와 관련된 파일 경로(DIFF 메시지만 해당)"""

    # SKILL message fields - only populated for SKILL messages
    skill_name: str | None = None
    """호출된 스킬의 이름입니다."""

    skill_description: str | None = None
    """스킬에 대한 간략한 설명입니다."""

    skill_source: str | None = None
    """스킬의 출처(예: `'built-in'`, `'user'`, `'project'`)."""

    skill_args: str | None = None
    """기술 호출에 대한 사용자 제공 인수입니다."""

    skill_body: str | None = None
    """전체 SKILL.md 콘텐츠가 에이전트에게 전송되었습니다."""

    skill_expanded: bool = False
    """스킬 본문이 UI에서 확장되는지 여부입니다."""

    is_streaming: bool = False
    """메시지가 아직 스트리밍되고 있는지 여부입니다.

    `True` 동안 해당 위젯은 콘텐츠 청크를 적극적으로 수신하므로 정리하거나 다시 하이드레이션해서는 안 됩니다.

    """

    height_hint: int | None = None
    """스크롤 위치 추정을 위해 터미널 행의 캐시된 위젯 높이입니다.

    `_hydrate_messages_above`이 뷰포트 위에 위젯을 삽입할 때 사용자의 뷰가 점프하지 않도록 스크롤 오프셋을 조정해야 합니다. 현재
    이는 고정 추정치(메시지당 5행)를 사용합니다. 첫 번째 마운트 후 여기에 실제 렌더링된 높이를 캐싱하면 특히 diff나 긴 어시스턴트 응답과 같은 긴
    메시지의 경우 예측이 정확해집니다.

    아직 채워지지 않았습니다. `app.py`의 `_hydrate_messages_above`을(를) 참조하세요.

    """

    def __post_init__(self) -> None:
        """구성 후 유형 필드 일관성을 검증합니다.

Raises:
            ValueError: TOOL 메시지에 `tool_name`이 없거나 SKILL 메시지에 `skill_name`이 누락된 경우.

        """
        if self.type == MessageType.TOOL and not self.tool_name:
            msg = "TOOL messages must have a tool_name"
            raise ValueError(msg)
        if self.type == MessageType.SKILL and not self.skill_name:
            msg = "SKILL messages must have a skill_name"
            raise ValueError(msg)

    def to_widget(self) -> Widget:
        """이 메시지 데이터에서 위젯을 다시 만듭니다.

Returns:
            이 데이터에 적합한 메시지 위젯입니다.

        """
        # Import here to avoid circular imports
        from deepagents_cli.widgets.messages import (
            AppMessage,
            AssistantMessage,
            DiffMessage,
            ErrorMessage,
            SkillMessage,
            SummarizationMessage,
            ToolCallMessage,
            UserMessage,
        )

        match self.type:
            case MessageType.USER:
                return UserMessage(self.content, id=self.id)

            case MessageType.ASSISTANT:
                return AssistantMessage(self.content, id=self.id)

            case MessageType.TOOL:
                widget = ToolCallMessage(
                    self.tool_name or "unknown",
                    self.tool_args,
                    id=self.id,
                )
                # Deferred state is restored automatically during on_mount
                # via _restore_deferred_state
                widget._deferred_status = self.tool_status
                widget._deferred_output = self.tool_output
                widget._deferred_expanded = self.tool_expanded
                return widget

            case MessageType.SKILL:
                widget = SkillMessage(
                    skill_name=self.skill_name or "unknown",
                    description=self.skill_description or "",
                    source=self.skill_source or "",
                    body=self.skill_body or "",
                    args=self.skill_args or "",
                    id=self.id,
                )
                widget._deferred_expanded = self.skill_expanded
                return widget

            case MessageType.ERROR:
                return ErrorMessage(self.content, id=self.id)

            case MessageType.APP:
                return AppMessage(self.content, id=self.id)

            case MessageType.SUMMARIZATION:
                return SummarizationMessage(self.content, id=self.id)

            case MessageType.DIFF:
                return DiffMessage(
                    self.content,
                    file_path=self.diff_file_path or "",
                    id=self.id,
                )

            case _:
                logger.warning(
                    "Unknown MessageType %r for message %s, falling back to AppMessage",
                    self.type,
                    self.id,
                )
                return AppMessage(self.content, id=self.id)

    @classmethod
    def from_widget(cls, widget: Widget) -> MessageData:
        """기존 위젯에서 MessageData를 만듭니다.

Args:
            widget: 직렬화할 메시지 위젯입니다.

Returns:
            모든 위젯의 상태를 포함하는 MessageData입니다.

        """
        # Deferred: prevents import-order issue — both modules live in the
        # widgets package, and messages is re-exported from widgets/__init__.
        from deepagents_cli.widgets.messages import (
            AppMessage,
            AssistantMessage,
            DiffMessage,
            ErrorMessage,
            SkillMessage,
            SummarizationMessage,
            ToolCallMessage,
            UserMessage,
        )

        widget_id = widget.id or f"msg-{uuid.uuid4().hex[:8]}"

        if isinstance(widget, SkillMessage):
            return cls(
                type=MessageType.SKILL,
                content="",
                id=widget_id,
                skill_name=widget._skill_name,
                skill_description=widget._description,
                skill_source=widget._source,
                skill_body=widget._body,
                skill_args=widget._args,
                skill_expanded=widget._expanded,
            )

        if isinstance(widget, UserMessage):
            return cls(
                type=MessageType.USER,
                content=widget._content,
                id=widget_id,
            )

        if isinstance(widget, AssistantMessage):
            return cls(
                type=MessageType.ASSISTANT,
                content=widget._content,
                id=widget_id,
                is_streaming=widget._stream is not None,
            )

        if isinstance(widget, ToolCallMessage):
            tool_status: ToolStatus | None = None
            if widget._status:
                try:
                    tool_status = ToolStatus(widget._status)
                except ValueError:
                    logger.warning(
                        "Unknown tool status %r for widget %s",
                        widget._status,
                        widget_id,
                    )

            return cls(
                type=MessageType.TOOL,
                content="",  # Tool messages don't have simple content
                id=widget_id,
                tool_name=widget._tool_name,
                tool_args=widget._args,
                tool_status=tool_status,
                tool_output=widget._output,
                tool_expanded=widget._expanded,
            )

        if isinstance(widget, ErrorMessage):
            return cls(
                type=MessageType.ERROR,
                content=widget._content,
                id=widget_id,
            )

        # Check specialized subclasses before AppMessage so we keep their type
        # when serializing and can restore their specific styling later.
        if isinstance(widget, DiffMessage):
            return cls(
                type=MessageType.DIFF,
                content=widget._diff_content,
                id=widget_id,
                diff_file_path=widget._file_path,
            )

        if isinstance(widget, SummarizationMessage):
            return cls(
                type=MessageType.SUMMARIZATION,
                content=str(widget._content),
                id=widget_id,
            )

        if isinstance(widget, AppMessage):
            return cls(
                type=MessageType.APP,
                content=str(widget._content),
                id=widget_id,
            )

        logger.warning(
            "Unknown widget type %s (id=%s), storing as APP message",
            type(widget).__name__,
            widget_id,
        )
        return cls(
            type=MessageType.APP,
            content=f"[Unknown widget: {type(widget).__name__}]",
            id=widget_id,
        )


class MessageStore:
    """가상화를 위한 메시지 데이터 및 위젯 창을 관리합니다.

    이 클래스는 모든 메시지를 데이터로 저장하고 실제로 DOM에 마운트되는 위젯의 슬라이딩 창을 관리합니다.

Attributes:
        WINDOW_SIZE: DOM에 보관할 최대 위젯 수입니다.

            부드러운 스크롤 경험과 DOM 성능의 균형을 유지합니다.
        HYDRATE_BUFFER: 가장자리 근처에서 스크롤할 때 하이드레이트할 메시지 수입니다.

            눈에 띄는 로딩 일시 중지를 방지할 만큼 충분한 버퍼를 제공합니다.

    """

    WINDOW_SIZE: int = 50
    HYDRATE_BUFFER: int = 15

    def __init__(self) -> None:
        """메시지 저장소를 초기화합니다."""
        self._messages: list[MessageData] = []
        self._visible_start: int = 0
        self._visible_end: int = 0

        # Track active streaming message - never archive this
        self._active_message_id: str | None = None

    @property
    def total_count(self) -> int:
        """저장된 총 메시지 수입니다."""
        return len(self._messages)

    @property
    def visible_count(self) -> int:
        """현재 표시되는 메시지 수(위젯)"""
        return self._visible_end - self._visible_start

    @property
    def has_messages_above(self) -> bool:
        """보이는 창 위에 보관된 메시지가 있는지 확인하세요."""
        return self._visible_start > 0

    @property
    def has_messages_below(self) -> bool:
        """보이는 창 아래에 보관된 메시지가 있는지 확인하세요."""
        return self._visible_end < len(self._messages)

    def append(self, message: MessageData) -> None:
        """스토어에 새 메시지를 추가합니다.

Args:
            message: 추가할 메시지 데이터입니다.

        """
        self._messages.append(message)
        self._visible_end = len(self._messages)

    def bulk_load(
        self, messages: list[MessageData]
    ) -> tuple[list[MessageData], list[MessageData]]:
        """한 번에 많은 메시지를 로드하고 꼬리 부분만 표시되도록 합니다.

        이는 스레드 재개에 최적화되어 있습니다. 모든 메시지는 경량 데이터로 저장되지만 마지막 `WINDOW_SIZE` 항목만 표시되는 것으로
        표시됩니다(즉, DOM 위젯이 필요함).

Args:
            messages: 로드할 메시지 데이터의 순서가 지정된 목록입니다.

Returns:
            (보관된, 표시되는) 메시지 목록의 튜플입니다.

        """
        self._messages.extend(messages)
        total = len(self._messages)

        if total <= self.WINDOW_SIZE:
            self._visible_start = 0
        else:
            self._visible_start = total - self.WINDOW_SIZE

        self._visible_end = total

        archived = self._messages[: self._visible_start]
        visible = self._messages[self._visible_start : self._visible_end]
        return archived, visible

    def get_message(self, message_id: str) -> MessageData | None:
        """해당 ID로 메시지를 받습니다.

Args:
            message_id: 찾을 메시지의 ID입니다.

Returns:
            메시지 데이터 또는 찾을 수 없는 경우 None입니다.

        """
        for msg in self._messages:
            if msg.id == message_id:
                return msg
        return None

    def get_message_at_index(self, index: int) -> MessageData | None:
        """색인으로 메시지를 가져옵니다.

Args:
            index: 메시지의 색인입니다.

Returns:
            메시지 데이터 또는 인덱스가 범위를 벗어난 경우 None입니다.

        """
        if 0 <= index < len(self._messages):
            return self._messages[index]
        return None

    def update_message(self, message_id: str, **updates: Any) -> bool:
        """메시지 데이터를 업데이트합니다.

        `_UPDATABLE_FIELDS`의 필드만 업데이트할 수 있습니다. 알 수 없는 필드 이름은 오타를 조기에 발견하기 위해
        `ValueError`을 발생시킵니다.

Args:
            message_id: 업데이트할 메시지의 ID입니다.
            **updates: 업데이트할 필드입니다.

Returns:
            메시지가 발견되어 업데이트된 경우 True입니다.

Raises:
            ValueError: `updates`의 키가 업데이트 가능한 허용 목록에 없는 경우.

        """
        unknown = set(updates) - _UPDATABLE_FIELDS
        if unknown:
            msg = f"Cannot update unknown or protected fields: {unknown}"
            raise ValueError(msg)

        for msg_data in self._messages:
            if msg_data.id == message_id:
                for key, value in updates.items():
                    setattr(msg_data, key, value)
                return True
        return False

    def set_active_message(self, message_id: str | None) -> None:
        """현재 활성(스트리밍) 메시지를 설정합니다.

        활성 메시지는 보관되지 않습니다.

Args:
            message_id: 활성 메시지의 ID 또는 삭제할 경우 None입니다.

        """
        self._active_message_id = message_id

    def is_active(self, message_id: str) -> bool:
        """메시지가 활성 스트리밍 메시지인지 확인하세요.

Args:
            message_id: 확인할 메시지 ID입니다.

Returns:
            활성 메시지인 경우 참입니다.

        """
        return message_id == self._active_message_id

    def window_exceeded(self) -> bool:
        """보이는 창이 최대 크기를 초과하는지 확인하세요.

Returns:
            일부 위젯을 정리해야 한다면 참입니다.

        """
        return self.visible_count > self.WINDOW_SIZE

    def get_messages_to_prune(self, count: int | None = None) -> list[MessageData]:
        """정리해야 할 가장 오래된 표시 메시지를 가져옵니다.

        표시된 창의 시작부터 연속적인 메시지 실행을 반환합니다. 표시되는 창에 공백이 생성되는 것을 방지하기 위해 활성 스트리밍 메시지에서
        중지합니다(DOM에서 저장소 상태를 비동기화함).

Args:
            count: 정리할 메시지 수 또는 WINDOW_SIZE로 돌아갈 만큼 정리할 메시지가 없습니다.

Returns:
            정리할 메시지 목록(위젯 제거)

        """
        if count is None:
            count = max(0, self.visible_count - self.WINDOW_SIZE)

        if count <= 0:
            return []

        to_prune: list[MessageData] = []
        idx = self._visible_start

        while len(to_prune) < count and idx < self._visible_end:
            msg = self._messages[idx]
            # Stop at the active message to keep the window contiguous
            if msg.id == self._active_message_id:
                break
            to_prune.append(msg)
            idx += 1

        return to_prune

    def mark_pruned(self, message_ids: list[str]) -> None:
        """메시지를 정리됨(위젯 제거됨)으로 표시합니다.

        창 앞의 연속된 정리된 메시지를 지나 `_visible_start` 전진합니다.

Args:
            message_ids: 정리된 메시지의 ID입니다.

        """
        pruned_set = set(message_ids)
        while (
            self._visible_start < self._visible_end
            and self._messages[self._visible_start].id in pruned_set
        ):
            self._visible_start += 1

    def get_messages_to_hydrate(self, count: int | None = None) -> list[MessageData]:
        """보이는 창 위에 메시지를 받아 수화하세요.

Args:
            count: 하이드레이트할 메시지 수 또는 `HYDRATE_BUFFER`에 대한 메시지 없음.

Returns:
            순서대로 하이드레이트(위젯 생성)할 메시지 목록입니다.

        """
        if count is None:
            count = self.HYDRATE_BUFFER

        if self._visible_start <= 0:
            return []

        hydrate_start = max(0, self._visible_start - count)
        return self._messages[hydrate_start : self._visible_start]

    def mark_hydrated(self, count: int) -> None:
        """위의 메시지가 수화되었음을 표시하세요.

Args:
            count: 하이드레이트된 메시지 수입니다.

        """
        self._visible_start = max(0, self._visible_start - count)

    def should_hydrate_above(
        self, scroll_position: float, viewport_height: int
    ) -> bool:
        """현재 보기 위에 메시지를 하이드레이션해야 하는지 확인하세요.

Args:
            scroll_position: 현재 스크롤 Y 위치.
            viewport_height: 뷰포트의 높이.

Returns:
            사용자가 상단 근처에서 스크롤하고 있고 메시지가 보관되어 있으면 참입니다.

        """
        if not self.has_messages_above:
            return False

        # Hydrate when within 2x viewport height of the top
        threshold = viewport_height * 2
        return scroll_position < threshold

    def should_prune_below(
        self, scroll_position: float, viewport_height: int, content_height: int
    ) -> bool:
        """현재 보기 아래의 메시지를 정리해야 하는지 확인하세요.

Note:
            아직 스크롤 핸들러에 통합되지 않았습니다. 사용자가 위로 스크롤할 때 나중에 뷰포트 아래에 있는 메시지를 정리하기 위한 것입니다.

Args:
            scroll_position: 현재 스크롤 Y 위치.
            viewport_height: 뷰포트의 높이.
            content_height: 모든 콘텐츠의 총 높이입니다.

Returns:
            위젯이 너무 많고 아래쪽 위젯이 시야에서 멀리 떨어져 있으면 참입니다.

        """
        if self.visible_count <= self.WINDOW_SIZE:
            return False

        # Only prune if user is far from the bottom
        distance_from_bottom = content_height - scroll_position - viewport_height
        threshold = viewport_height * 3
        return distance_from_bottom > threshold

    def clear(self) -> None:
        """모든 메시지를 지웁니다."""
        self._messages.clear()
        self._visible_start = 0
        self._visible_end = 0
        self._active_message_id = None

    def get_visible_range(self) -> tuple[int, int]:
        """표시되는 메시지 색인의 범위를 가져옵니다.

Returns:
            (start_index, end_index)의 튜플입니다.

        """
        return (self._visible_start, self._visible_end)

    def get_all_messages(self) -> list[MessageData]:
        """저장된 모든 메시지를 가져옵니다.

Returns:
            모든 메시지 데이터 목록(얕은 복사본)입니다.

        """
        return list(self._messages)

    def get_visible_messages(self) -> list[MessageData]:
        """보이는 창에서 메시지를 받으세요.

Returns:
            표시되는 메시지 데이터 목록입니다.

        """
        return self._messages[self._visible_start : self._visible_end]
