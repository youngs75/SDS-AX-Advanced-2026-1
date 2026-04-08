"""LangGraph 실행 이벤트를 텍스트 UI 업데이트에 연결합니다.

이 어댑터는 스트리밍된 모델/도구 이벤트를 메시지 저장소 업데이트, 인터럽트 위젯, 토큰 계정 및 Textual 앱이 점진적으로 렌더링할 수 있는 상태
변경으로 변환합니다.
"""
# This module has complex streaming logic ported from execution.py

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path
    from typing import Protocol

    from langchain.agents.middleware.human_in_the_loop import (
        ApproveDecision,
        EditDecision,
        HITLRequest,
        RejectDecision,
    )
    from langchain_core.messages import AIMessage
    from langchain_core.runnables import RunnableConfig
    from langgraph.types import Command, Interrupt
    from pydantic import TypeAdapter
    from rich.console import Console

    from deepagents_cli._ask_user_types import AskUserWidgetResult, Question

    # Type alias matching HITLResponse["decisions"] element type
    HITLDecision = ApproveDecision | EditDecision | RejectDecision

    class _TokensUpdateCallback(Protocol):
        """`_on_tokens_update`에 대한 콜백 서명입니다."""

        def __call__(self, count: int, *, approximate: bool = False) -> None: ...

    class _TokensShowCallback(Protocol):
        """`_on_tokens_show`에 대한 콜백 서명입니다."""

        def __call__(self, *, approximate: bool = False) -> None: ...


from deepagents_cli._ask_user_types import AskUserRequest
from deepagents_cli._cli_context import CLIContext  # noqa: TC001
from deepagents_cli._debug import configure_debug_logging
from deepagents_cli._session_stats import (
    ModelStats as ModelStats,
    SessionStats as SessionStats,
    SpinnerStatus as SpinnerStatus,
    format_token_count as format_token_count,
)
from deepagents_cli.config import build_stream_config
from deepagents_cli.file_ops import FileOpTracker
from deepagents_cli.formatting import format_duration
from deepagents_cli.hooks import dispatch_hook
from deepagents_cli.input import MediaTracker, parse_file_mentions
from deepagents_cli.media_utils import create_multimodal_content
from deepagents_cli.tool_display import format_tool_message_content
from deepagents_cli.widgets.messages import (
    AppMessage,
    AssistantMessage,
    DiffMessage,
    SummarizationMessage,
    ToolCallMessage,
)

logger = logging.getLogger(__name__)
configure_debug_logging(logger)

_hitl_adapter_cache: TypeAdapter | None = None
"""HITL 요청 유효성 검사기에 대한 게으른 싱글톤입니다."""


# ---------------------------------------------------------------------------
# Adapter-local schema helpers
# ---------------------------------------------------------------------------

def _get_hitl_request_adapter(hitl_request_type: type) -> TypeAdapter:
    """캐시된 `TypeAdapter(HITLRequest)`을(를) 반환합니다.

    `execute_task_textual` 호출마다 pydantic 스키마를 다시 컴파일하는 것을 방지합니다.

    Args:
        hitl_request_type: `HITLRequest` 클래스(호출자가 로컬로 가져오기 때문에 전달됨)

    Returns:
        `TypeAdapter` 인스턴스를 공유했습니다.

    """
    global _hitl_adapter_cache  # noqa: PLW0603
    if _hitl_adapter_cache is None:
        from pydantic import TypeAdapter

        _hitl_adapter_cache = TypeAdapter(hitl_request_type)
    return _hitl_adapter_cache


def print_usage_table(
    stats: SessionStats,
    wall_time: float,
    console: Console,
) -> None:
    """모델 사용 통계 테이블을 Rich 콘솔에 인쇄합니다.

    세션이 여러 모델에 걸쳐 있는 경우 각 모델에는 합계 행이 추가된 자체 행이 있습니다. 단일 모델 세션은 하나의 행을 표시합니다.

    Args:
        stats: 누적 세션 통계입니다.
        wall_time: 총 벽시계 시간(초)입니다.
        console: 출력을 위한 풍부한 콘솔.

    """
    from rich.table import Table

    has_time = wall_time >= 0.1  # noqa: PLR2004
    if not (stats.request_count or stats.input_tokens or has_time):
        return

    if stats.per_model:
        multi_model = len(stats.per_model) > 1

        table = Table(
            show_header=True,
            header_style="bold",
            box=None,
            padding=(0, 2, 0, 0),
            show_edge=False,
        )
        table.add_column("Model", style="dim")
        table.add_column("Reqs", justify="right", style="dim")
        table.add_column("InputTok", justify="right", style="dim")
        table.add_column("OutputTok", justify="right", style="dim")

        if multi_model:
            for model_name, ms in stats.per_model.items():
                table.add_row(
                    model_name,
                    str(ms.request_count),
                    format_token_count(ms.input_tokens),
                    format_token_count(ms.output_tokens),
                )
            table.add_row(
                "Total",
                str(stats.request_count),
                format_token_count(stats.input_tokens),
                format_token_count(stats.output_tokens),
            )
        else:
            model_label = next(iter(stats.per_model))
            table.add_row(
                model_label,
                str(stats.request_count),
                format_token_count(stats.input_tokens),
                format_token_count(stats.output_tokens),
            )

        console.print()
        console.print("[bold]Usage Stats[/bold]")
        console.print(table)
    if has_time:
        console.print()
        console.print(
            f"Agent active  {format_duration(wall_time)}",
            style="dim",
            highlight=False,
        )


_ask_user_adapter_cache: TypeAdapter | None = None
"""`ask_user` 인터럽트 유효성 검사기에 대한 게으른 싱글톤입니다."""


def _get_ask_user_adapter() -> TypeAdapter:
    """캐시된 `TypeAdapter(AskUserRequest)`을(를) 반환합니다.

    Returns:
        `TypeAdapter` 인스턴스를 공유했습니다.

    """
    global _ask_user_adapter_cache  # noqa: PLW0603
    if _ask_user_adapter_cache is None:
        from pydantic import TypeAdapter

        _ask_user_adapter_cache = TypeAdapter(AskUserRequest)
    return _ask_user_adapter_cache


def _is_summarization_chunk(metadata: dict | None) -> bool:
    """메시지 청크가 요약 미들웨어에서 나온 것인지 확인하세요.

    요약 모델은 LangChain의 콜백 시스템이 스트림 메타데이터 사전에 병합하는 `config={"metadata": {"lc_source":
    "summarization"}}`(`langchain.agents.middleware.summarization` 참조)으로 호출됩니다.

    Args:
        metadata: 스트림 청크의 메타데이터 dict입니다.

    Returns:
        청크가 요약에서 나온 것이며 필터링되어야 하는지 여부입니다.

    """
    if metadata is None:
        return False
    return metadata.get("lc_source") == "summarization"


# ---------------------------------------------------------------------------
# Textual-facing bridge object
# ---------------------------------------------------------------------------

class TextualUIAdapter:
    """에이전트 출력을 텍스트 위젯으로 렌더링하기 위한 어댑터입니다.

    이 어댑터는 에이전트 실행과 텍스트 UI 사이에 추상화 계층을 제공하여 스트리밍 출력을 위젯으로 렌더링할 수 있도록 합니다.

    """

    def __init__(
        self,
        mount_message: Callable[..., Awaitable[None]],
        update_status: Callable[[str], None],
        request_approval: Callable[..., Awaitable[Any]],
        on_auto_approve_enabled: Callable[[], None] | None = None,
        set_spinner: Callable[[SpinnerStatus], Awaitable[None]] | None = None,
        set_active_message: Callable[[str | None], None] | None = None,
        sync_message_content: Callable[[str, str], None] | None = None,
        request_ask_user: (
            Callable[
                [list[Question]],
                Awaitable[asyncio.Future[AskUserWidgetResult] | None],
            ]
            | None
        ) = None,
    ) -> None:
        """어댑터를 초기화합니다."""
        self._mount_message = mount_message
        """메시지 위젯을 채팅에 마운트하기 위한 비동기 콜백."""

        self._update_status = update_status
        """상태 표시줄 텍스트를 업데이트하는 콜백입니다."""

        self._request_approval = request_approval
        """HITL 승인을 위해 Future를 반환하는 비동기 콜백입니다."""

        self._on_auto_approve_enabled = on_auto_approve_enabled
        """HITL 승인을 통해 자동 승인이 활성화되면 콜백이 호출됩니다.
        메뉴.

        사용자가 승인 대화 상자에서 '모두 자동 승인'을 선택하면 실행되어 앱이 상태 표시줄과 세션 상태를 동기화할 수 있습니다.

        """

        self._set_spinner = set_spinner
        """로딩 스피너를 표시/숨기기 위한 콜백입니다."""

        self._set_active_message = set_active_message
        """활성 스트리밍 메시지 ID를 설정하기 위한 콜백(지우려면 `None` 전달)"""

        self._sync_message_content = sync_message_content
        """스트리밍 후 최종 메시지 콘텐츠를 스토어에 다시 동기화하기 위한 콜백입니다."""

        self._request_ask_user = request_ask_user
        """`ask_user` 인터럽트에 대한 비동기 콜백.

        기다렸다가 사용자 답변을 확인하는 `Future`을 반환합니다.

        """

        # State tracking
        self._current_tool_messages: dict[str, ToolCallMessage] = {}
        """도구 호출 ID를 해당 메시지 위젯에 매핑합니다."""

        # Token display callbacks (set by the app after construction)
        self._on_tokens_update: _TokensUpdateCallback | None = None
        """각 LLM 응답 후 전체 컨텍스트 토큰으로 호출됩니다."""

        self._on_tokens_hide: Callable[[], None] | None = None
        """스트리밍 중에 토큰 표시를 숨기기 위해 호출됩니다."""

        self._on_tokens_show: _TokensShowCallback | None = None
        """캐시된 값으로 토큰 표시를 복원하기 위해 호출됩니다."""

    def finalize_pending_tools_with_error(self, error: str) -> None:
        """보류 중/실행 중인 모든 도구 위젯을 오류 및 명확한 추적으로 표시합니다.

        이는 일치하는 `ToolMessage` 결과가 수신되기 전에 예기치 않은 예외로 인해 스트리밍이 중단될 때 안전망으로 사용됩니다.

        Args:
            error: 보류 중인 각 도구 위젯에 표시할 오류 텍스트입니다.

        """
        for tool_msg in list(self._current_tool_messages.values()):
            tool_msg.set_error(error)
        self._current_tool_messages.clear()

        # Clear active streaming message to avoid stale "active" state in the store.
        if self._set_active_message:
            self._set_active_message(None)


def _build_interrupted_ai_message(
    pending_text_by_namespace: dict[tuple, str],
    current_tool_messages: dict[str, Any],
) -> AIMessage | None:
    """중단된 상태(텍스트 + 도구 호출)를 캡처하는 AIMessage를 빌드합니다.

    Args:
        pending_text_by_namespace: 네임스페이스별 누적 텍스트 사전
        current_tool_messages: tool_id 사전 -> ToolCallMessage 위젯

    Returns:
        누적된 콘텐츠 및 도구 호출이 포함된 AIMessage 또는 비어 있는 경우 None입니다.

    """
    from langchain_core.messages import AIMessage

    main_ns_key = ()
    accumulated_text = pending_text_by_namespace.get(main_ns_key, "").strip()

    # Reconstruct tool_calls from displayed tool messages
    tool_calls = []
    for tool_id, tool_widget in list(current_tool_messages.items()):
        tool_calls.append(
            {
                "id": tool_id,
                "name": tool_widget._tool_name,
                "args": tool_widget._args,
            }
        )

    if not accumulated_text and not tool_calls:
        return None

    return AIMessage(
        content=accumulated_text,
        tool_calls=tool_calls or [],
    )


def _read_mentioned_file(file_path: Path, max_embed_bytes: int) -> str:
    """인라인 임베딩(동기화, to_thread와 함께 사용)을 위해 언급된 파일을 읽습니다.

    Args:
        file_path: 파일의 확인된 경로입니다.
        max_embed_bytes: 크기 임계값 더 큰 파일은 참조만 얻습니다.

    Returns:
        파일 콘텐츠 또는 크기 초과 참조가 포함된 마크다운 스니펫입니다.

    """
    file_size = file_path.stat().st_size
    if file_size > max_embed_bytes:
        size_kb = file_size // 1024
        return (
            f"\n### {file_path.name}\n"
            f"Path: `{file_path}`\n"
            f"Size: {size_kb}KB (too large to embed, "
            "use read_file tool to view)"
        )
    content = file_path.read_text(encoding="utf-8")
    return f"\n### {file_path.name}\nPath: `{file_path}`\n```\n{content}\n```"


# ---------------------------------------------------------------------------
# Stream execution and post-stream cleanup
# ---------------------------------------------------------------------------

async def execute_task_textual(
    user_input: str,
    agent: Any,  # noqa: ANN401  # Dynamic agent graph type
    assistant_id: str | None,
    session_state: Any,  # noqa: ANN401  # Dynamic session state type
    adapter: TextualUIAdapter,
    backend: Any = None,  # noqa: ANN401  # Dynamic backend type
    image_tracker: MediaTracker | None = None,
    context: CLIContext | None = None,
    *,
    sandbox_type: str | None = None,
    message_kwargs: dict[str, Any] | None = None,
    turn_stats: SessionStats | None = None,
) -> SessionStats:
    """텍스트 UI로 전달되는 출력으로 작업을 실행합니다.

    이는 모든 UI 작업에 TextualUIAdapter를 사용하는 Execute_task()의 텍스트 호환 버전입니다.

    Args:
        user_input: 사용자의 입력 메시지
        agent: 실행할 LangGraph 에이전트
        assistant_id: 에이전트 식별자
        session_state: auto_approve 플래그가 있는 세션 상태
        adapter: UI 작업을 위한 TextualUIAdapter
        backend: 파일 작업을 위한 선택적 백엔드
        image_tracker: 이미지용 선택적 추적기
        context: 모델 재정의 및 매개변수가 포함된 선택적 `CLIContext`은 `context=`을 통해 그래프에 전달됩니다.
        sandbox_type: 추적 메타데이터에 대한 샌드박스 제공자 이름 또는 활성화된 샌드박스가 없는 경우 `None`입니다.
        message_kwargs: 추가 필드가 스트림 입력 메시지 dict에 병합되었습니다(예: 체크포인트에서 기술 메타데이터를 유지하기 위한
                        `additional_kwargs`).
        turn_stats: 축적할 `SessionStats`이(가) 미리 생성되었습니다.

            호출자가 동일한 객체에 대한 참조를 보유하면 이 코루틴이 반환되기 전에 취소되더라도 통계를 사용할 수 있습니다.

            `None`인 경우 새 인스턴스가 내부적으로 생성됩니다.

    Returns:
        이번 턴 동안 누적된 통계(요청 수, 토큰 수,
            벽시계 시간).

    Raises:
        ValidationError: HITL 요청 유효성 검사가 실패하는 경우(다시 발생)

    """
    from langchain.agents.middleware.human_in_the_loop import (
        ApproveDecision,
        HITLRequest,
        RejectDecision,
    )
    from langchain_core.messages import HumanMessage, ToolMessage
    from langgraph.types import Command
    from pydantic import ValidationError

    hitl_request_adapter = _get_hitl_request_adapter(HITLRequest)
    ask_user_adapter = _get_ask_user_adapter()

    # Parse file mentions and inject content if any — offload blocking I/O
    prompt_text, mentioned_files = await asyncio.to_thread(
        parse_file_mentions, user_input
    )

    # Max file size to embed inline (256KB, matching mistral-vibe)
    # Larger files get a reference instead - use read_file tool to view them
    max_embed_bytes = 256 * 1024

    if mentioned_files:
        context_parts = [prompt_text, "\n\n## Referenced Files\n"]
        for file_path in mentioned_files:
            try:
                part = await asyncio.to_thread(
                    _read_mentioned_file, file_path, max_embed_bytes
                )
                context_parts.append(part)
            except Exception as e:  # noqa: BLE001  # Resilient adapter error handling
                context_parts.append(
                    f"\n### {file_path.name}\n[Error reading file: {e}]"
                )
        final_input = "\n".join(context_parts)
    else:
        final_input = prompt_text

    # Include images and videos in the message content
    images_to_send = []
    videos_to_send = []
    if image_tracker:
        images_to_send = image_tracker.get_images()
        videos_to_send = image_tracker.get_videos()
    if images_to_send or videos_to_send:
        message_content = create_multimodal_content(
            final_input, images_to_send, videos_to_send
        )
    else:
        message_content = final_input

    thread_id = session_state.thread_id
    config = build_stream_config(thread_id, assistant_id, sandbox_type=sandbox_type)

    await dispatch_hook("session.start", {"thread_id": thread_id})

    captured_input_tokens = 0
    captured_output_tokens = 0
    if turn_stats is None:
        turn_stats = SessionStats()
    start_time = time.monotonic()

    # Warn if token display callbacks are only partially wired — all three
    # should be set together to avoid inconsistent status-bar behavior.
    token_cbs = (
        adapter._on_tokens_update,
        adapter._on_tokens_hide,
        adapter._on_tokens_show,
    )
    if any(token_cbs) and not all(token_cbs):
        logger.warning(
            "Token callbacks partially wired (update=%s, hide=%s, show=%s); "
            "token display may behave inconsistently",
            adapter._on_tokens_update is not None,
            adapter._on_tokens_hide is not None,
            adapter._on_tokens_show is not None,
        )

    # Show spinner
    if adapter._set_spinner:
        await adapter._set_spinner("Thinking")

    # Hide token display during streaming (will be shown with accurate count at end)
    if adapter._on_tokens_hide:
        adapter._on_tokens_hide()

    file_op_tracker = FileOpTracker(assistant_id=assistant_id, backend=backend)
    displayed_tool_ids: set[str] = set()
    tool_call_buffers: dict[str | int, dict] = {}

    # Track pending text and assistant messages PER NAMESPACE to avoid interleaving
    # when multiple subagents stream in parallel
    pending_text_by_namespace: dict[tuple, str] = {}
    assistant_message_by_namespace: dict[tuple, Any] = {}

    # Clear media from tracker after creating the message
    if image_tracker:
        image_tracker.clear()

    user_msg: dict[str, Any] = {"role": "user", "content": message_content}
    if message_kwargs:
        user_msg.update(message_kwargs)
    stream_input: dict | Command = {"messages": [user_msg]}

    # Track summarization lifecycle so spinner status and notification stay in sync.
    summarization_in_progress = False

    try:
        while True:
            interrupt_occurred = False
            suppress_resumed_output = False
            pending_interrupts: dict[str, HITLRequest] = {}
            pending_ask_user: dict[str, AskUserRequest] = {}

            async for chunk in agent.astream(
                stream_input,
                stream_mode=["messages", "updates"],
                subgraphs=True,
                config=config,
                context=context,
                durability="exit",
            ):
                if not isinstance(chunk, tuple) or len(chunk) != 3:  # noqa: PLR2004  # stream chunk is a 3-tuple (namespace, mode, data)
                    logger.debug("Skipping non-3-tuple chunk: %s", type(chunk).__name__)
                    continue

                namespace, current_stream_mode, data = chunk

                # Convert namespace to hashable tuple for dict keys
                ns_key = tuple(namespace) if namespace else ()

                # Filter out subagent outputs - only show main agent (empty
                # namespace). Subagents run via Task tool and should only
                # report back to the main agent
                is_main_agent = ns_key == ()

                # Handle UPDATES stream - for interrupts and todos
                if current_stream_mode == "updates":
                    if not isinstance(data, dict):
                        continue

                    # Check for interrupts
                    if "__interrupt__" in data:
                        interrupts: list[Interrupt] = data["__interrupt__"]
                        if interrupts:
                            for interrupt_obj in interrupts:
                                iv = interrupt_obj.value
                                if (
                                    isinstance(iv, dict)
                                    and iv.get("type") == "ask_user"
                                ):
                                    try:
                                        validated_ask_user = (
                                            ask_user_adapter.validate_python(iv)
                                        )
                                        pending_ask_user[interrupt_obj.id] = (
                                            validated_ask_user
                                        )
                                        interrupt_occurred = True
                                        await dispatch_hook("input.required", {})
                                    except ValidationError:
                                        logger.exception(
                                            "Invalid ask_user interrupt payload"
                                        )
                                        raise
                                else:
                                    try:
                                        validated_request = (
                                            hitl_request_adapter.validate_python(iv)
                                        )
                                        pending_interrupts[interrupt_obj.id] = (
                                            validated_request
                                        )
                                        interrupt_occurred = True
                                        await dispatch_hook("input.required", {})
                                    except ValidationError:  # noqa: TRY203  # Re-raise preserves exception context in handler
                                        raise

                    # Check for todo updates (not yet implemented in Textual UI)
                    chunk_data = next(iter(data.values())) if data else None
                    if (
                        chunk_data
                        and isinstance(chunk_data, dict)
                        and "todos" in chunk_data
                    ):
                        pass  # Future: render todo list widget

                # Handle MESSAGES stream - for content and tool calls
                elif current_stream_mode == "messages":
                    # Skip subagent outputs - only render main agent content in chat
                    if not is_main_agent:
                        logger.debug("Skipping subagent message ns=%s", ns_key)
                        continue

                    if not isinstance(data, tuple) or len(data) != 2:  # noqa: PLR2004  # message stream data is a 2-tuple (message, metadata)
                        logger.debug(
                            "Skipping non-2-tuple message data: type=%s",
                            type(data).__name__,
                        )
                        continue

                    message, metadata = data
                    logger.debug(
                        "Processing message: type=%s id=%s has_content_blocks=%s",
                        type(message).__name__,
                        getattr(message, "id", None),
                        hasattr(message, "content_blocks"),
                    )

                    # Filter out summarization model output, but keep UI feedback.
                    # The summarization model streams AIMessage chunks tagged
                    # with lc_source="summarization" in the callback metadata.
                    # These are hidden from the user; only the spinner and a
                    # notification widget provide feedback.
                    if _is_summarization_chunk(metadata):
                        if not summarization_in_progress:
                            summarization_in_progress = True
                            if adapter._set_spinner:
                                await adapter._set_spinner("Offloading")
                        continue

                    # Regular (non-summarization) chunks resumed — summarization
                    # has finished. Mount the notification and reset the spinner.
                    if summarization_in_progress:
                        summarization_in_progress = False
                        try:
                            await adapter._mount_message(SummarizationMessage())
                        except Exception:
                            logger.debug(
                                "Failed to mount summarization notification",
                                exc_info=True,
                            )
                        if adapter._set_spinner and not adapter._current_tool_messages:
                            await adapter._set_spinner("Thinking")

                    if isinstance(message, HumanMessage):
                        content = message.text
                        # Flush pending text for this namespace
                        pending_text = pending_text_by_namespace.get(ns_key, "")
                        if content and pending_text:
                            await _flush_assistant_text_ns(
                                adapter,
                                pending_text,
                                ns_key,
                                assistant_message_by_namespace,
                            )
                            pending_text_by_namespace[ns_key] = ""
                        continue

                    if isinstance(message, ToolMessage):
                        tool_name = getattr(message, "name", "")
                        tool_status = getattr(message, "status", "success")
                        tool_content = format_tool_message_content(message.content)
                        record = file_op_tracker.complete_with_message(message)

                        # Update tool call status with output
                        tool_id = getattr(message, "tool_call_id", None)
                        if tool_id and tool_id in adapter._current_tool_messages:
                            # Pop before widget calls so the dict drains even
                            # if set_success/set_error raises.
                            tool_msg = adapter._current_tool_messages.pop(tool_id)
                            output_str = str(tool_content) if tool_content else ""
                            if tool_status == "success":
                                tool_msg.set_success(output_str)
                            else:
                                tool_msg.set_error(output_str or "Error")
                                await dispatch_hook(
                                    "tool.error",
                                    {"tool_names": [tool_msg._tool_name]},
                                )
                        elif tool_id:
                            logger.debug(
                                "ToolMessage tool_call_id=%s not in "
                                "_current_tool_messages; spinner gating "
                                "may be stale",
                                tool_id,
                            )

                        # Reshow spinner only when all in-flight tools have
                        # completed (avoids premature "Thinking..." when
                        # parallel tool calls are active).
                        if adapter._set_spinner and not adapter._current_tool_messages:
                            await adapter._set_spinner("Thinking")

                        # Show file operation results - always show diffs in chat
                        if record:
                            pending_text = pending_text_by_namespace.get(ns_key, "")
                            if pending_text:
                                await _flush_assistant_text_ns(
                                    adapter,
                                    pending_text,
                                    ns_key,
                                    assistant_message_by_namespace,
                                )
                                pending_text_by_namespace[ns_key] = ""
                            if record.diff:
                                await adapter._mount_message(
                                    DiffMessage(record.diff, record.display_path)
                                )
                        continue

                    # Extract token usage (before content_blocks check
                    # - usage may be on any chunk)
                    if hasattr(message, "usage_metadata"):
                        usage = message.usage_metadata
                        if usage:
                            input_toks = usage.get("input_tokens", 0)
                            output_toks = usage.get("output_tokens", 0)
                            total_toks = usage.get("total_tokens", 0)
                            from deepagents_cli.config import settings

                            active_model = settings.model_name or ""
                            if input_toks or output_toks:
                                # Model gives split counts — preferred path
                                turn_stats.record_request(
                                    active_model, input_toks, output_toks
                                )
                                captured_input_tokens = max(
                                    captured_input_tokens, input_toks + output_toks
                                )
                            elif total_toks:
                                # Fallback: model gives only total (no split)
                                turn_stats.record_request(active_model, total_toks, 0)
                                captured_input_tokens = max(
                                    captured_input_tokens, total_toks
                                )

                    # Check if this is an AIMessageChunk with content
                    if not hasattr(message, "content_blocks"):
                        logger.debug(
                            "Message has no content_blocks: type=%s",
                            type(message).__name__,
                        )
                        continue

                    # Process content blocks
                    blocks = message.content_blocks
                    logger.debug(
                        "content_blocks count=%d blocks=%s",
                        len(blocks),
                        repr(blocks)[:500],
                    )
                    for block in blocks:
                        block_type = block.get("type")

                        if block_type == "text":
                            text = block.get("text", "")
                            if text:
                                # Track accumulated text for reference
                                pending_text = pending_text_by_namespace.get(ns_key, "")
                                pending_text += text
                                pending_text_by_namespace[ns_key] = pending_text

                                # Get or create assistant message for this namespace
                                current_msg = assistant_message_by_namespace.get(ns_key)
                                if current_msg is None:
                                    # Hide spinner when assistant starts responding
                                    if adapter._set_spinner:
                                        await adapter._set_spinner(None)
                                    msg_id = f"asst-{uuid.uuid4().hex[:8]}"
                                    # Mark active BEFORE mounting so pruning
                                    # (triggered by mount) won't remove it
                                    # (_mount_message can trigger
                                    # _prune_old_messages if the window exceeds
                                    # WINDOW_SIZE.)
                                    if adapter._set_active_message:
                                        adapter._set_active_message(msg_id)
                                    current_msg = AssistantMessage(id=msg_id)
                                    await adapter._mount_message(current_msg)
                                    assistant_message_by_namespace[ns_key] = current_msg

                                # Append just the new text chunk for smoother
                                # streaming (uses MarkdownStream internally for
                                # better performance)
                                await current_msg.append_content(text)

                        elif block_type in {"tool_call_chunk", "tool_call"}:
                            chunk_name = block.get("name")
                            chunk_args = block.get("args")
                            chunk_id = block.get("id")
                            chunk_index = block.get("index")

                            buffer_key: str | int
                            if chunk_index is not None:
                                buffer_key = chunk_index
                            elif chunk_id is not None:
                                buffer_key = chunk_id
                            else:
                                buffer_key = f"unknown-{len(tool_call_buffers)}"

                            buffer = tool_call_buffers.setdefault(
                                buffer_key,
                                {
                                    "name": None,
                                    "id": None,
                                    "args": None,
                                    "args_parts": [],
                                },
                            )

                            if chunk_name:
                                buffer["name"] = chunk_name
                            if chunk_id:
                                buffer["id"] = chunk_id

                            if isinstance(chunk_args, dict):
                                buffer["args"] = chunk_args
                                buffer["args_parts"] = []
                            elif isinstance(chunk_args, str):
                                if chunk_args:
                                    parts: list[str] = buffer.setdefault(
                                        "args_parts", []
                                    )
                                    if not parts or chunk_args != parts[-1]:
                                        parts.append(chunk_args)
                                    buffer["args"] = "".join(parts)
                            elif chunk_args is not None:
                                buffer["args"] = chunk_args

                            buffer_name = buffer.get("name")
                            buffer_id = buffer.get("id")
                            if buffer_name is None:
                                continue

                            parsed_args = buffer.get("args")
                            if isinstance(parsed_args, str):
                                if not parsed_args:
                                    continue
                                try:
                                    parsed_args = json.loads(parsed_args)
                                except json.JSONDecodeError:
                                    continue
                            elif parsed_args is None:
                                continue

                            if not isinstance(parsed_args, dict):
                                parsed_args = {"value": parsed_args}

                            # Flush pending text before tool call
                            pending_text = pending_text_by_namespace.get(ns_key, "")
                            if pending_text:
                                await _flush_assistant_text_ns(
                                    adapter,
                                    pending_text,
                                    ns_key,
                                    assistant_message_by_namespace,
                                )
                                pending_text_by_namespace[ns_key] = ""
                                assistant_message_by_namespace.pop(ns_key, None)

                            logger.debug(
                                "Tool call buffer: name=%s id=%s args=%s",
                                buffer_name,
                                buffer_id,
                                repr(parsed_args)[:200],
                            )
                            if (
                                buffer_id is not None
                                and buffer_id not in displayed_tool_ids
                            ):
                                displayed_tool_ids.add(buffer_id)
                                file_op_tracker.start_operation(
                                    buffer_name, parsed_args, buffer_id
                                )

                                # Hide spinner before showing tool call
                                if adapter._set_spinner:
                                    await adapter._set_spinner(None)

                                # Mount tool call message
                                logger.debug(
                                    "Mounting ToolCallMessage: %s(%s)",
                                    buffer_name,
                                    repr(parsed_args)[:200],
                                )
                                tool_msg = ToolCallMessage(buffer_name, parsed_args)
                                await adapter._mount_message(tool_msg)
                                adapter._current_tool_messages[buffer_id] = tool_msg

                            tool_call_buffers.pop(buffer_key, None)

                    if getattr(message, "chunk_position", None) == "last":
                        pending_text = pending_text_by_namespace.get(ns_key, "")
                        if pending_text:
                            await _flush_assistant_text_ns(
                                adapter,
                                pending_text,
                                ns_key,
                                assistant_message_by_namespace,
                            )
                            pending_text_by_namespace[ns_key] = ""
                            assistant_message_by_namespace.pop(ns_key, None)

            # Reset summarization state if stream ended mid-summarization
            # (e.g. middleware error, stream exhausted before regular chunks).
            if summarization_in_progress:
                summarization_in_progress = False
                try:
                    await adapter._mount_message(SummarizationMessage())
                except Exception:
                    logger.debug(
                        "Failed to mount summarization notification",
                        exc_info=True,
                    )
                if adapter._set_spinner and not adapter._current_tool_messages:
                    await adapter._set_spinner("Thinking")

            # Flush any remaining text from all namespaces
            for ns_key, pending_text in list(pending_text_by_namespace.items()):
                if pending_text:
                    await _flush_assistant_text_ns(
                        adapter, pending_text, ns_key, assistant_message_by_namespace
                    )
            pending_text_by_namespace.clear()
            assistant_message_by_namespace.clear()

            # Handle HITL after stream completes
            if interrupt_occurred:
                any_rejected = False
                resume_payload: dict[str, Any] = {}

                for interrupt_id, ask_req in list(pending_ask_user.items()):
                    questions = ask_req["questions"]

                    if adapter._request_ask_user:
                        if adapter._set_spinner:
                            await adapter._set_spinner(None)
                        result: dict[str, Any] = {
                            "type": "error",
                            "error": "ask_user callback returned no response",
                        }
                        try:
                            future = await adapter._request_ask_user(questions)
                        except Exception:
                            logger.exception("Failed to mount ask_user widget")
                            result = {
                                "type": "error",
                                "error": "failed to display ask_user prompt",
                            }
                            future = None

                        if future is None:
                            logger.error(
                                "ask_user callback returned no Future; "
                                "reporting as error"
                            )
                        else:
                            try:
                                future_result = await future
                                if isinstance(future_result, dict):
                                    result = future_result
                                else:
                                    logger.error(
                                        "ask_user future returned non-dict result: %s",
                                        type(future_result).__name__,
                                    )
                                    result = {
                                        "type": "error",
                                        "error": "invalid ask_user widget result",
                                    }
                            except Exception:
                                logger.exception(
                                    "ask_user future resolution failed; "
                                    "reporting as error"
                                )
                                result = {
                                    "type": "error",
                                    "error": "failed to receive ask_user response",
                                }

                        result_type = result.get("type")
                        if result_type == "answered":
                            answers = result.get("answers", [])
                            if isinstance(answers, list):
                                resume_payload[interrupt_id] = {"answers": answers}
                                tool_id = ask_req["tool_call_id"]
                                if tool_id in adapter._current_tool_messages:
                                    tool_msg = adapter._current_tool_messages[tool_id]
                                    tool_msg.set_success("User answered")
                                    adapter._current_tool_messages.pop(tool_id, None)
                            else:
                                logger.error(
                                    "ask_user answered payload had non-list "
                                    "answers: %s",
                                    type(answers).__name__,
                                )
                                resume_payload[interrupt_id] = {
                                    "status": "error",
                                    "error": "invalid ask_user answers payload",
                                    "answers": ["" for _ in questions],
                                }
                                any_rejected = True
                        elif result_type == "cancelled":
                            resume_payload[interrupt_id] = {
                                "status": "cancelled",
                                "answers": ["" for _ in questions],
                            }
                            any_rejected = True
                        else:
                            error_text = result.get("error")
                            if not isinstance(error_text, str) or not error_text:
                                error_text = "ask_user interaction failed"
                            resume_payload[interrupt_id] = {
                                "status": "error",
                                "error": error_text,
                                "answers": ["" for _ in questions],
                            }
                            any_rejected = True
                    else:
                        logger.warning(
                            "ask_user interrupt received but no UI callback is "
                            "registered; reporting as error"
                        )
                        resume_payload[interrupt_id] = {
                            "status": "error",
                            "error": "ask_user not supported by this UI",
                            "answers": ["" for _ in questions],
                        }

                for interrupt_id, hitl_request in list(pending_interrupts.items()):
                    action_requests = hitl_request["action_requests"]

                    if session_state.auto_approve:
                        decisions: list[HITLDecision] = [
                            ApproveDecision(type="approve") for _ in action_requests
                        ]
                        resume_payload[interrupt_id] = {"decisions": decisions}
                        for tool_msg in list(adapter._current_tool_messages.values()):
                            tool_msg.set_running()
                    else:
                        # Batch approval - one dialog for all parallel tool calls
                        await dispatch_hook(
                            "permission.request",
                            {
                                "tool_names": [
                                    r.get("name", "") for r in action_requests
                                ]
                            },
                        )
                        future = await adapter._request_approval(
                            action_requests, assistant_id
                        )
                        decision = await future

                        if isinstance(decision, dict):
                            decision_type = decision.get("type")

                            if decision_type == "auto_approve_all":
                                session_state.auto_approve = True
                                if adapter._on_auto_approve_enabled:
                                    adapter._on_auto_approve_enabled()
                                decisions = [
                                    ApproveDecision(type="approve")
                                    for _ in action_requests
                                ]
                                tool_msgs = list(
                                    adapter._current_tool_messages.values()
                                )
                                for tool_msg in tool_msgs:
                                    tool_msg.set_running()
                                for action_request in action_requests:
                                    tool_name = action_request.get("name")
                                    if tool_name in {
                                        "write_file",
                                        "edit_file",
                                    }:
                                        args = action_request.get("args", {})
                                        if isinstance(args, dict):
                                            file_op_tracker.mark_hitl_approved(
                                                tool_name, args
                                            )

                            elif decision_type == "approve":
                                decisions = [
                                    ApproveDecision(type="approve")
                                    for _ in action_requests
                                ]
                                tool_msgs = list(
                                    adapter._current_tool_messages.values()
                                )
                                for tool_msg in tool_msgs:
                                    tool_msg.set_running()
                                for action_request in action_requests:
                                    tool_name = action_request.get("name")
                                    if tool_name in {
                                        "write_file",
                                        "edit_file",
                                    }:
                                        args = action_request.get("args", {})
                                        if isinstance(args, dict):
                                            file_op_tracker.mark_hitl_approved(
                                                tool_name, args
                                            )

                            elif decision_type == "reject":
                                decisions = [
                                    RejectDecision(type="reject")
                                    for _ in action_requests
                                ]
                                tool_msgs = list(
                                    adapter._current_tool_messages.values()
                                )
                                for tool_msg in tool_msgs:
                                    tool_msg.set_rejected()
                                adapter._current_tool_messages.clear()
                                any_rejected = True
                            else:
                                logger.warning(
                                    "Unexpected HITL decision type: %s",
                                    decision_type,
                                )
                                decisions = [
                                    RejectDecision(type="reject")
                                    for _ in action_requests
                                ]
                                for tool_msg in list(
                                    adapter._current_tool_messages.values()
                                ):
                                    tool_msg.set_rejected()
                                adapter._current_tool_messages.clear()
                                any_rejected = True
                        else:
                            logger.warning(
                                "HITL decision was not a dict: %s",
                                type(decision).__name__,
                            )
                            decisions = [
                                RejectDecision(type="reject") for _ in action_requests
                            ]
                            for tool_msg in list(
                                adapter._current_tool_messages.values()
                            ):
                                tool_msg.set_rejected()
                            adapter._current_tool_messages.clear()
                            any_rejected = True

                        resume_payload[interrupt_id] = {"decisions": decisions}

                        if any_rejected:
                            break

                suppress_resumed_output = any_rejected

            if interrupt_occurred and resume_payload:
                if suppress_resumed_output and not pending_ask_user:
                    await adapter._mount_message(
                        AppMessage(
                            "Command rejected. Tell the agent what you'd like instead."
                        )
                    )
                    turn_stats.wall_time_seconds = time.monotonic() - start_time
                    return turn_stats

                stream_input = Command(resume=resume_payload)
            else:
                await dispatch_hook("task.complete", {"thread_id": thread_id})
                break

    except (asyncio.CancelledError, KeyboardInterrupt):
        await _handle_interrupt_cleanup(
            adapter=adapter,
            agent=agent,
            config=config,
            pending_text_by_namespace=pending_text_by_namespace,
            captured_input_tokens=captured_input_tokens,
            captured_output_tokens=captured_output_tokens,
            turn_stats=turn_stats,
            start_time=start_time,
        )
        return turn_stats

    # Update token count and return stats
    turn_stats.wall_time_seconds = time.monotonic() - start_time
    await _report_and_persist_tokens(
        adapter,
        agent,
        config,
        captured_input_tokens,
        captured_output_tokens,
    )
    return turn_stats


async def _handle_interrupt_cleanup(
    *,
    adapter: TextualUIAdapter,
    agent: Any,  # noqa: ANN401  # Dynamic agent graph type
    config: RunnableConfig,
    pending_text_by_namespace: dict[tuple, str],
    captured_input_tokens: int,
    captured_output_tokens: int,
    turn_stats: SessionStats,
    start_time: float,
) -> None:
    """CancelledError 및 KeyboardInterrupt에 대한 공유 정리입니다.

    Args:
        adapter: 디스플레이 콜백이 포함된 UI 어댑터.
        agent: LangGraph 에이전트.
        config: `thread_id`으로 실행 가능한 구성입니다.
        pending_text_by_namespace: 네임스페이스당 누적된 텍스트입니다.
        captured_input_tokens: 인터럽트 전에 캡처된 입력 토큰입니다.
        captured_output_tokens: 인터럽트 전에 캡처된 출력 토큰입니다.
        turn_stats: 현재 턴의 통계입니다.
        start_time: 턴이 시작되었을 때의 단조로운 타임스탬프입니다.

    """
    from langchain_core.messages import HumanMessage

    # Clear active message immediately so it won't block pruning.
    # If we don't do this, the store still thinks it's active and protects
    # from pruning, which breaks get_messages_to_prune(), potentially
    # blocking all future pruning.
    if adapter._set_active_message:
        adapter._set_active_message(None)

    # Hide spinner (may still show "Offloading" if interrupted mid-offload)
    if adapter._set_spinner:
        await adapter._set_spinner(None)

    await adapter._mount_message(AppMessage("Interrupted by user"))

    interrupted_msg = _build_interrupted_ai_message(
        pending_text_by_namespace,
        adapter._current_tool_messages,
    )

    # Save accumulated state before marking tools as rejected (best-effort).
    # State update failures shouldn't prevent cleanup.
    try:
        if interrupted_msg:
            await agent.aupdate_state(config, {"messages": [interrupted_msg]})

        cancellation_msg = HumanMessage(
            content="[SYSTEM] Task interrupted by user. "
            "Previous operation was cancelled."
        )
        await agent.aupdate_state(config, {"messages": [cancellation_msg]})
    except Exception:
        logger.warning("Failed to save interrupted state", exc_info=True)

    # Mark tools as rejected AFTER saving state
    for tool_msg in list(adapter._current_tool_messages.values()):
        tool_msg.set_rejected()
    adapter._current_tool_messages.clear()

    # Keep the token count marked stale whenever interrupted state was captured,
    # including tool-only turns after assistant text was already flushed.
    approximate = interrupted_msg is not None

    turn_stats.wall_time_seconds = time.monotonic() - start_time
    await _report_and_persist_tokens(
        adapter,
        agent,
        config,
        captured_input_tokens,
        captured_output_tokens,
        shield=True,
        approximate=approximate,
    )


async def _persist_context_tokens(
    agent: Any,  # noqa: ANN401  # Dynamic agent graph type
    config: RunnableConfig,
    tokens: int,
) -> None:
    """컨텍스트 토큰 수를 그래프 상태로 유지하는 최선의 노력입니다.

    Args:
        agent: LangGraph 에이전트(`aupdate_state`을 지원해야 함)
        config: `thread_id`으로 실행 가능한 구성입니다.
        tokens: 유지할 총 컨텍스트 토큰입니다.

    """
    try:
        await agent.aupdate_state(config, {"_context_tokens": tokens})
    except Exception:  # non-critical; stale count on resume is acceptable
        logger.warning(
            "Failed to persist _context_tokens=%d; token count may be stale on resume",
            tokens,
            exc_info=True,
        )


async def _report_and_persist_tokens(
    adapter: TextualUIAdapter,
    agent: Any,  # noqa: ANN401  # Dynamic agent graph type
    config: RunnableConfig,
    captured_input_tokens: int,
    captured_output_tokens: int,
    *,
    shield: bool = False,
    approximate: bool = False,
) -> None:
    """토큰 표시를 업데이트하고 최선을 다해 그래프 상태를 유지합니다.

    Args:
        adapter: 토큰 콜백이 포함된 UI 어댑터.
        agent: LangGraph 에이전트.
        config: 구성 가능한 사전에 `thread_id`을 사용하여 실행 가능한 구성입니다.
        captured_input_tokens: 해당 턴 동안 캡처된 총 입력 토큰입니다.
        captured_output_tokens: 해당 턴 동안 캡처된 총 출력 토큰입니다.
        shield: `True`인 경우 인터럽트 핸들러가 이를 안전하게 기다릴 수 있도록 지속 호출에서 예외 및 `CancelledError`을
                억제합니다.
        approximate: `True`인 경우 "+"를 추가하여 개수가 오래되었음을 UI에 알립니다(예: 생성이 중단된 후).

    """
    if captured_input_tokens or captured_output_tokens:
        if adapter._on_tokens_update:
            adapter._on_tokens_update(captured_input_tokens, approximate=approximate)
        if shield:
            try:
                await _persist_context_tokens(agent, config, captured_input_tokens)
            except (Exception, asyncio.CancelledError):
                logger.debug(
                    "Token persist suppressed during interrupt cleanup",
                    exc_info=True,
                )
        else:
            await _persist_context_tokens(agent, config, captured_input_tokens)
    elif adapter._on_tokens_show:
        adapter._on_tokens_show(approximate=approximate)


async def _flush_assistant_text_ns(
    adapter: TextualUIAdapter,
    text: str,
    ns_key: tuple,
    assistant_message_by_namespace: dict[tuple, Any],
) -> None:
    """특정 네임스페이스에 대해 누적된 보조 텍스트를 플러시합니다.

    MarkdownStream을 중지하여 스트리밍을 마무리합니다. 아직 메시지가 없으면 전체 콘텐츠가 포함된 메시지를 만듭니다.

    """
    if not text.strip():
        return

    current_msg = assistant_message_by_namespace.get(ns_key)
    if current_msg is None:
        # No message was created during streaming - create one with full content
        msg_id = f"asst-{uuid.uuid4().hex[:8]}"
        current_msg = AssistantMessage(text, id=msg_id)
        await adapter._mount_message(current_msg)
        await current_msg.write_initial_content()
        assistant_message_by_namespace[ns_key] = current_msg
    else:
        # Stop the stream to finalize the content
        await current_msg.stop_stream()

    # When the AssistantMessage was first mounted and recorded in the
    # MessageStore, it had empty content (streaming hadn't started yet).
    # Now that streaming is done, the widget holds the full text in
    # `_content`, but the store's MessageData still has `content=""`.
    # If the message is later pruned and re-hydrated, `to_widget()` would
    # recreate it from that stale empty string. This call copies the
    # widget's final content back into the store so re-hydration works.
    if adapter._sync_message_content and current_msg.id:
        adapter._sync_message_content(current_msg.id, current_msg._content)

    # Clear active message since streaming is done
    if adapter._set_active_message:
        adapter._set_active_message(None)
