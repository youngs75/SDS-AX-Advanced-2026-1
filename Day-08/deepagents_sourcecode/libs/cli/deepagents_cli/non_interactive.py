"""deepagents CLI의 비대화형 실행 모드입니다.

에이전트 그래프에 대해 단일 사용자 작업을 실행하고 결과를 stdout으로 스트리밍하고 적절한 코드로 종료하는 `run_non_interactive`을
제공합니다.

에이전트는 `RemoteAgent` 클라이언트를 통해 연결된 `langgraph dev` 서버 하위 프로세스 내에서
실행됩니다(`server_manager.server_session` 참조).

셸 명령은 선택적 허용 목록(`--shell-allow-list`)에 의해 관리됩니다.

- 설정되지 않음 → 쉘이 비활성화되고 다른 모든 도구 호출은 자동 승인됩니다. - `recommended` 또는 명시적 목록 → 쉘 활성화, 명령 검증
    목록에 반대; 비쉘 도구는 무조건 승인됩니다.
- `all` → 쉘 활성화, 모든 명령 허용, 모든 도구 자동 승인.

선택적 자동 모드(`--quiet` / `-q`)는 모든 콘솔 출력을 stderr로 리디렉션하고 stdout은 에이전트의 응답 텍스트용으로만 남겨둡니다.
"""


from __future__ import annotations

import logging
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from langchain.agents.middleware.human_in_the_loop import ActionRequest, HITLRequest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command, Interrupt
from pydantic import TypeAdapter, ValidationError
from rich.console import Console
from rich.live import Live
from rich.markup import escape as escape_markup
from rich.spinner import Spinner as RichSpinner
from rich.style import Style
from rich.text import Text

from deepagents_cli._version import __version__
from deepagents_cli.agent import DEFAULT_AGENT_NAME
from deepagents_cli.config import (
    SHELL_ALLOW_ALL,
    SHELL_TOOL_NAMES,
    build_langsmith_thread_url,
    create_model,
    is_shell_command_allowed,
    settings,
)
from deepagents_cli.file_ops import FileOpTracker
from deepagents_cli.hooks import dispatch_hook, dispatch_hook_fire_and_forget
from deepagents_cli.model_config import ModelConfigError
from deepagents_cli.sessions import generate_thread_id
from deepagents_cli.textual_adapter import SessionStats, print_usage_table
from deepagents_cli.unicode_security import (
    check_url_safety,
    detect_dangerous_unicode,
    format_warning_detail,
    iter_string_values,
    looks_like_url_key,
    summarize_issues,
)

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig

logger = logging.getLogger(__name__)


class HITLIterationLimitError(RuntimeError):
    """HITL 인터럽트 루프가 `_MAX_HITL_ITERATIONS` 라운드를 초과하면 발생합니다."""



_HITL_REQUEST_ADAPTER = TypeAdapter(HITLRequest)

_STREAM_CHUNK_LENGTH = 3
"""Agent.astream이 내보낸 튜플의 예상 요소 수입니다.

스트림 청크는 3개의 튜플(네임스페이스, stream_mode, 데이터)입니다.
"""


_MESSAGE_DATA_LENGTH = 2
"""메시지 모드 데이터는 2-튜플(message_obj, 메타데이터)입니다."""


_MAX_HITL_ITERATIONS = 50
"""무한 루프를 방지하기 위해 HITL 인터럽트 왕복 횟수에 대한 안전 제한(예: 에이전트가 거부된 명령을 계속 재시도하는 경우)"""



def _write_text(text: str) -> None:
    """stdout에 에이전트 응답 텍스트를 씁니다(후행 줄 바꿈 없이).

    콘솔이 stderr로 리디렉션되는 자동 모드에서도 에이전트 응답 텍스트가 항상 stdout에 표시되도록 `sys.stdout`을(Rich Console
    대신) 직접 사용합니다.

    Args:
        text: 작성할 텍스트 문자열입니다.

    """

    sys.stdout.write(text)
    sys.stdout.flush()


def _write_newline() -> None:
    """stdout에 개행 문자를 쓰고 플러시합니다."""

    sys.stdout.write("\n")
    sys.stdout.flush()


class _ConsoleSpinner:
    """비대화형 자세한 출력을 위한 애니메이션 스피너입니다.

    중지되면 사라지는 임시 점자 회전 기능이 있는 Rich의 `Live` 디스플레이를 사용하여 터미널 출력을 깔끔하게 유지합니다.

    """


    def __init__(self, console: Console) -> None:
        self._console = console
        self._live: Live | None = None

    def start(self, message: str = "Working...") -> None:
        """주어진 메시지로 스피너를 시작하십시오.

        스피너가 이미 실행 중이면 작동하지 않습니다. 콘솔이 라이브 디스플레이를 지원할 수 없으면 자동으로 실패합니다.

        Args:
            message: 스피너 옆에 표시할 상태 텍스트입니다.

        """

        if self._live is not None:
            return
        renderable = RichSpinner(
            "dots",
            text=Text(f" {message}", style="dim"),
            style="dim",
        )
        try:
            self._live = Live(renderable, console=self._console, transient=True)
            self._live.start()
        except (AttributeError, TypeError, OSError) as exc:
            logger.warning("Spinner start failed: %s", exc)
            self._live = None

    def stop(self) -> None:
        """실행 중인 경우 스피너를 중지합니다. `start`로 다시 시작할 수 있습니다."""

        if self._live is not None:
            try:
                self._live.stop()
            except (AttributeError, TypeError, OSError) as exc:
                logger.warning("Spinner stop failed: %s", exc)
            finally:
                self._live = None


@dataclass
class StreamState:
    """에이전트 스트림을 반복하는 동안 변경 가능한 상태가 누적되었습니다."""


    quiet: bool = False
    """`True`인 경우 표준 출력으로 이동되는 진단 형식
    (예: 도구 알림 앞의 구분 기호 줄 바꿈)은 stdout에 에이전트 응답 텍스트만 포함되도록 억제됩니다.
    """


    stream: bool = True
    """`True`(기본값)이면 텍스트 청크가 도착하자마자 stdout에 기록됩니다.

    `False`인 경우 텍스트는 `full_response`에 버퍼링되고 에이전트가 완료된 후 플러시됩니다.

    """


    full_response: list[str] = field(default_factory=list)
    """AI 메시지 스트림에서 누적된 텍스트 조각입니다."""


    tool_call_buffers: dict[int | str, dict[str, str | None]] = field(
        default_factory=dict
    )
    """진행 중인 도구 호출 색인 또는 ID를 해당 이름/ID 메타데이터에 매핑합니다.
    도구 호출.
    """


    pending_interrupts: dict[str, HITLRequest] = field(default_factory=dict)
    """대기 중인 검증된 HITL 요청에 인터럽트 ID를 매핑합니다.
    결정.
    """


    hitl_response: dict[str, dict[str, list[dict[str, str]]]] = field(
        default_factory=dict
    )
    """인터럽트 ID를 다음 목록과 함께 `'decisions'` 키가 포함된 사전에 매핑합니다.
    결정 명령(각각 `'approve'` 또는 `'reject'`의 `'type'` 키를 가짐)

    HITL 처리 후 에이전트를 재개하는 데 사용됩니다.

    """


    interrupt_occurred: bool = False
    """HITL 인터럽트가 수신되었는지 여부를 나타내는 플래그입니다.
    현재 스트림 패스.
    """


    stats: SessionStats = field(default_factory=SessionStats)
    """이 스트림에 대해 누적된 모델 사용 통계입니다."""


    spinner: _ConsoleSpinner | None = None
    """자세한 정보 표시 모드에서 에이전트 작업 중에 표시되는 선택적 애니메이션 스피너입니다."""



@dataclass
class ThreadUrlLookupState:
    """최선의 노력을 기울이는 백그라운드 LangSmith 스레드 URL 조회 상태입니다.

    스레드 안전성: 백그라운드 스레드는 `url`을 설정한 다음 `done.set()`을 호출합니다. 소비자는 `url`을 읽기 전에
    `done.is_set()`을 확인해야 합니다.

    """


    done: threading.Event = field(default_factory=threading.Event)
    url: str | None = None


def _start_langsmith_thread_url_lookup(thread_id: str) -> ThreadUrlLookupState:
    """차단하지 않고 백그라운드 LangSmith URL 확인을 시작합니다.

    Args:
        thread_id: 해결할 스레드 식별자입니다.

    Returns:
        나중에 완료를 확인할 수 있는 변경 가능한 조회 상태입니다.

    """

    state = ThreadUrlLookupState()

    def _resolve() -> None:
        try:
            state.url = build_langsmith_thread_url(thread_id)
        except Exception:  # build_langsmith_thread_url already handles known errors
            logger.debug(
                "Could not resolve LangSmith thread URL for '%s'",
                thread_id,
                exc_info=True,
            )
        finally:
            state.done.set()

    threading.Thread(target=_resolve, daemon=True).start()
    return state


def _process_interrupts(
    data: dict[str, list[Interrupt]],
    state: StreamState,
    console: Console,
) -> None:
    """`updates` 청크에서 HITL 인터럽트를 추출하고 기록합니다.

    Args:
        data: `__interrupt__` 키가 포함된 `updates` 사전입니다.
        state: 새로운 보류 중인 인터럽트로 업데이트할 상태를 스트리밍합니다.
        console: 사용자에게 표시되는 경고를 위한 풍부한 콘솔.

    """

    interrupts = data["__interrupt__"]
    if interrupts:
        for interrupt_obj in interrupts:
            try:
                validated_request = _HITL_REQUEST_ADAPTER.validate_python(
                    interrupt_obj.value
                )
            except ValidationError:
                logger.warning(
                    "Rejecting malformed HITL interrupt %s (raw value: %r)",
                    interrupt_obj.id,
                    interrupt_obj.value,
                )
                console.print(
                    f"[yellow]Warning: Received malformed tool approval "
                    f"request (interrupt {interrupt_obj.id}). Rejecting.[/yellow]"
                )
                # Fail-closed: record a reject decision for malformed interrupts

                state.hitl_response[interrupt_obj.id] = {
                    "decisions": [{"type": "reject", "message": "Malformed interrupt"}]
                }
                continue
            state.pending_interrupts[interrupt_obj.id] = validated_request
            state.interrupt_occurred = True
            dispatch_hook_fire_and_forget("input.required", {})


def _process_ai_message(
    message_obj: AIMessage,
    state: StreamState,
    console: Console,
) -> None:
    """AI 메시지에서 텍스트 및 도구 호출 블록을 추출하고 렌더링합니다.

    스트리밍이 활성화되면 텍스트 블록이 즉시 stdout에 기록됩니다. 그렇지 않으면 지연된 출력을 위해 `state.full_response`에
    누적됩니다. 도구 호출 블록은 버퍼링되고 해당 이름이 콘솔에 인쇄됩니다.

    Args:
        message_obj: 스트림에서 수신된 `AIMessage`입니다.
        state: 응답 텍스트 및 도구 호출 버퍼를 축적하기 위한 스트림 상태입니다.
        console: 형식화된 출력을 위한 풍부한 콘솔.

    """

    # Extract token usage for stats accumulation
    usage = getattr(message_obj, "usage_metadata", None)
    if usage:
        input_toks = usage.get("input_tokens", 0)
        output_toks = usage.get("output_tokens", 0)
        total_toks = usage.get("total_tokens", 0)
        active_model = settings.model_name or ""
        if input_toks or output_toks:
            state.stats.record_request(active_model, input_toks, output_toks)
        elif total_toks:
            state.stats.record_request(active_model, total_toks, 0)

    if not hasattr(message_obj, "content_blocks"):
        logger.debug("AIMessage missing content_blocks attribute, skipping")
        return
    for block in message_obj.content_blocks:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text", "")
            if text:
                if state.stream:
                    if state.spinner:
                        state.spinner.stop()
                    _write_text(text)
                state.full_response.append(text)
        elif block_type in {"tool_call_chunk", "tool_call"}:
            chunk_name = block.get("name")
            chunk_id = block.get("id")
            chunk_index = block.get("index")

            if chunk_index is not None:
                buffer_key: int | str = chunk_index
            elif chunk_id is not None:
                buffer_key = chunk_id
            else:
                buffer_key = f"unknown-{len(state.tool_call_buffers)}"

            if buffer_key not in state.tool_call_buffers:
                state.tool_call_buffers[buffer_key] = {"name": None, "id": None}
            if chunk_name:
                state.tool_call_buffers[buffer_key]["name"] = chunk_name
                if state.spinner:
                    state.spinner.stop()
                if state.full_response and not state.quiet:
                    _write_newline()
                console.print(
                    f"[dim]🔧 Calling tool: {escape_markup(chunk_name)}[/dim]",
                    highlight=False,
                )


def _process_message_chunk(
    data: tuple[AIMessage | ToolMessage, dict[str, str]],
    state: StreamState,
    console: Console,
    file_op_tracker: FileOpTracker,
) -> None:
    """스트림에서 `messages` 모드 청크를 처리합니다.

    메시지 유형에 따라 AI 메시지 또는 도구 메시지 처리로 전달됩니다.

    Args:
        data: 메시지 스트림 모드의 `(message_obj, metadata)` 2튜플.
        state: 공유 스트림 상태.
        console: 형식화된 출력을 위한 풍부한 콘솔.
        file_op_tracker: 파일 작업 차이에 대한 추적기입니다.

    """

    if not isinstance(data, tuple) or len(data) != _MESSAGE_DATA_LENGTH:
        logger.debug(
            "Unexpected message-mode data (type=%s), skipping", type(data).__name__
        )
        return

    message_obj, metadata = data

    # The summarization middleware injects synthetic messages to compress
    # conversation history for the LLM. These are internal bookkeeping and
    # should not be rendered to the user.
    if metadata and metadata.get("lc_source") == "summarization":
        return

    if isinstance(message_obj, AIMessage):
        _process_ai_message(message_obj, state, console)
    elif isinstance(message_obj, ToolMessage):
        record = file_op_tracker.complete_with_message(message_obj)
        if record and record.diff:
            if state.spinner:
                state.spinner.stop()
            console.print(
                f"[dim]📝 {escape_markup(record.display_path)}[/dim]",
                highlight=False,
            )
        if state.spinner:
            state.spinner.start()


def _process_stream_chunk(
    chunk: object,
    state: StreamState,
    console: Console,
    file_op_tracker: FileOpTracker,
) -> None:
    """단일 원시 스트림 청크를 적절한 핸들러로 라우팅합니다.

    주 에이전트 청크만 처리됩니다. 최상위 콘텐츠만 렌더링되도록 하위 에이전트 출력이 무시됩니다.

    Args:
        chunk: `agent.astream`에서 생성된 원시 요소입니다.

            주 에이전트 출력의 경우 3튜플 `(namespace, stream_mode, data)`이 될 것으로 예상됩니다.
        state: 공유 스트림 상태.
        console: 형식화된 출력을 위한 풍부한 콘솔.
        file_op_tracker: 파일 작업 차이에 대한 추적기입니다.

    """

    if not isinstance(chunk, tuple) or len(chunk) != _STREAM_CHUNK_LENGTH:
        logger.debug(
            "Unexpected stream chunk (type=%s), skipping", type(chunk).__name__
        )
        return

    namespace, stream_mode, data = chunk
    is_main_agent = not namespace

    if not is_main_agent:
        return

    if stream_mode == "updates" and isinstance(data, dict) and "__interrupt__" in data:
        _process_interrupts(cast("dict[str, list[Interrupt]]", data), state, console)
    elif stream_mode == "messages":
        _process_message_chunk(
            cast("tuple[AIMessage | ToolMessage, dict[str, str]]", data),
            state,
            console,
            file_op_tracker,
        )


def _make_hitl_decision(
    action_request: ActionRequest, console: Console
) -> dict[str, str]:
    """단일 작업 요청을 승인할지 거부할지 결정합니다.

    이 함수는 제한적인 셸 허용 목록이 구성된 경우에만 호출됩니다(`all` 아님). 쉘이 비활성화되거나 제한되지 않으면 `interrupt_on`은 비어
    있고 이 기능은 완전히 우회됩니다.

    셸 도구는 항상 제한됩니다. 허용 목록이 구성된 경우 명령은 이에 대해 유효성이 검사됩니다. 허용 목록이 구성되지 않은 경우 셸 명령은 완전히
    거부됩니다(심층 방어 — 허용 목록이 없을 때 호출자는 셸 도구를 비활성화해야 하지만 이 기능은 관계없이 실패합니다). 비쉘 도구는 무조건 승인됩니다.

    Args:
        action_request: HITL 미들웨어가 내보낸 작업 요청 사전입니다.

            최소한 `name` 키를 포함해야 합니다.
        console: 상태 출력을 위한 풍부한 콘솔.

    Returns:
        `type` 키(`"approve"` 또는 `"reject"`)를 사용한 결정 dict
            사람이 읽을 수 있는 설명이 포함된 선택적 `message` 키.

    """

    for warning in _collect_action_request_warnings(action_request):
        console.print(f"[yellow]Warning:[/yellow] {warning}")

    action_name = action_request.get("name", "")

    if action_name in SHELL_TOOL_NAMES:
        if not settings.shell_allow_list:
            command = action_request.get("args", {}).get("command", "")
            console.print(
                f"\n[red]Shell command rejected (no allow-list configured): "
                f"{command}[/red]"
            )
            return {
                "type": "reject",
                "message": (
                    "Shell commands are not permitted in non-interactive mode "
                    "without a --shell-allow-list. Use --shell-allow-list to "
                    "specify allowed commands."
                ),
            }

        command = action_request.get("args", {}).get("command", "")

        if is_shell_command_allowed(command, settings.shell_allow_list):
            console.print(f"[dim]✓ Auto-approved: {escape_markup(command)}[/dim]")
            return {"type": "approve"}

        allowed_list_str = ", ".join(settings.shell_allow_list)
        console.print(f"\n[red]Shell command rejected:[/red] {escape_markup(command)}")
        console.print(
            f"[yellow]Allowed commands:[/yellow] {escape_markup(allowed_list_str)}"
        )
        return {
            "type": "reject",
            "message": (
                f"Command '{command}' is not in the allow-list. "
                f"Allowed commands: {allowed_list_str}. "
                f"Please use allowed commands or try another approach."
            ),
        }

    console.print(f"[dim]✓ Auto-approved action: {escape_markup(action_name)}[/dim]")
    return {"type": "approve"}


def _collect_action_request_warnings(action_request: ActionRequest) -> list[str]:
    """하나의 작업 요청에 대한 유니코드/URL 안전 경고를 수집합니다.

    작업 인수에 중첩된 모든 문자열 값을 재귀적으로 검사합니다.

    Returns:
        작업 인수의 의심스러운 값에 대한 경고 메시지입니다.

    """

    warnings: list[str] = []
    args = action_request.get("args", {})
    if not isinstance(args, dict):
        return warnings

    tool_name = str(action_request.get("name", "unknown"))

    for arg_path, text in iter_string_values(args):
        issues = detect_dangerous_unicode(text)
        if issues:
            warnings.append(
                f"{tool_name}.{arg_path} contains hidden Unicode "
                f"({summarize_issues(issues)})"
            )

        if looks_like_url_key(arg_path):
            safety = check_url_safety(text)
            if safety.safe:
                continue
            detail = format_warning_detail(safety.warnings)
            if safety.decoded_domain:
                detail = f"{detail}; decoded host: {safety.decoded_domain}"
            warnings.append(f"{tool_name}.{arg_path} URL warning: {detail}")

    return warnings


def _process_hitl_interrupts(state: StreamState, console: Console) -> None:
    """보류 중인 HITL 인터럽트를 반복하고 승인/거부 응답을 빌드합니다.

    처리 후에는 `state.pending_interrupts`이 지워지고 결정 사항이 `state.hitl_response`에 기록되므로 에이전트를
    재개할 수 있습니다.

    Args:
        state: 처리할 보류 중인 인터럽트가 포함된 스트림 상태입니다.
        console: 상태 출력을 위한 풍부한 콘솔.

    """

    current_interrupts = dict(state.pending_interrupts)
    state.pending_interrupts.clear()

    for interrupt_id, hitl_request in current_interrupts.items():
        decisions = [
            _make_hitl_decision(action_request, console)
            for action_request in hitl_request["action_requests"]
        ]
        state.hitl_response[interrupt_id] = {"decisions": decisions}


async def _stream_agent(
    agent: Any,  # noqa: ANN401
    stream_input: dict[str, Any] | Command,
    config: RunnableConfig,
    state: StreamState,
    console: Console,
    file_op_tracker: FileOpTracker,
) -> None:
    """전체 에이전트 스트림을 사용하고 결과로 *상태*를 업데이트합니다.

    Args:
        agent: 에이전트(Pregel 또는 RemoteAgent).
        stream_input: 초기 사용자 메시지 dict 또는 HITL 연속을 위한 `Command(resume=...)`입니다.
        config: LangGraph 실행 가능 구성(스레드 ID, 메타데이터 등)
        state: 공유 스트림 상태.
        console: 형식화된 출력을 위한 풍부한 콘솔.
        file_op_tracker: 파일 작업 차이에 대한 추적기입니다.

    """

    if state.spinner:
        state.spinner.start()
    try:
        async for chunk in agent.astream(
            stream_input,
            stream_mode=["messages", "updates"],
            subgraphs=True,
            config=config,
            durability="exit",
        ):
            _process_stream_chunk(chunk, state, console, file_op_tracker)
    finally:
        if state.spinner:
            state.spinner.stop()


async def _run_agent_loop(
    agent: Any,  # noqa: ANN401
    message: str,
    config: RunnableConfig,
    console: Console,
    file_op_tracker: FileOpTracker,
    *,
    quiet: bool = False,
    stream: bool = True,
    thread_url_lookup: ThreadUrlLookupState | None = None,
) -> None:
    """에이전트를 실행하고 작업이 완료될 때까지 HITL 인터럽트를 처리합니다.

    루프는 폭주 재시도를 방지하기 위해 최대 `_MAX_HITL_ITERATIONS` 라운드를 처리합니다(예: 에이전트가 거부된 명령을 반복적으로 시도하는
    경우).

    Args:
        agent: 에이전트(Pregel 또는 RemoteAgent).
        message: 사용자의 작업 메시지입니다.
        config: LangGraph 실행 가능 구성.
        console: 형식화된 출력을 위한 풍부한 콘솔.
        file_op_tracker: 파일 작업 차이에 대한 추적기입니다.
        quiet: stdout에서 진단 형식을 억제합니다.
        stream: `True`이면 텍스트가 도착하자마자 stdout에 기록됩니다.

            `False`인 경우 전체 응답이 버퍼링되고 마지막에 플러시됩니다.
        thread_url_lookup: Fast-Follow LangSmith 스레드 링크를 렌더링하기 위한 선택적 비차단 조회 상태입니다.

    Raises:
        HITLIterationLimitError: HITL 반복 제한이 초과된 경우.

    """

    spinner = None if quiet else _ConsoleSpinner(console)
    state = StreamState(quiet=quiet, stream=stream, spinner=spinner)
    stream_input: dict[str, Any] | Command = {
        "messages": [{"role": "user", "content": message}]
    }

    thread_id = config.get("configurable", {}).get("thread_id", "")
    await dispatch_hook("session.start", {"thread_id": thread_id})

    start_time = time.monotonic()

    # Initial stream
    await _stream_agent(agent, stream_input, config, state, console, file_op_tracker)

    # Handle HITL interrupts
    iterations = 0
    while state.interrupt_occurred:
        iterations += 1
        if iterations > _MAX_HITL_ITERATIONS:
            msg = (
                f"Exceeded {_MAX_HITL_ITERATIONS} HITL interrupt rounds. "
                "The agent may be stuck retrying rejected commands."
            )
            raise HITLIterationLimitError(msg)
        state.interrupt_occurred = False
        state.hitl_response.clear()
        _process_hitl_interrupts(state, console)
        stream_input = Command(resume=state.hitl_response)
        await _stream_agent(
            agent, stream_input, config, state, console, file_op_tracker
        )

    wall_time = time.monotonic() - start_time

    if state.full_response:
        if not state.stream:
            _write_text("".join(state.full_response))
        _write_newline()

    if not quiet:
        console.print()
        if (
            thread_url_lookup is not None
            and thread_url_lookup.done.is_set()
            and thread_url_lookup.url
        ):
            link_text = Text("View in LangSmith: ", style="dim")
            link_text.append(
                thread_url_lookup.url,
                style=Style(dim=True, link=thread_url_lookup.url),
            )
            console.print(link_text)
        console.print("[green]✓ Task completed[/green]")
        print_usage_table(state.stats, wall_time, console)

    await dispatch_hook("task.complete", {"thread_id": thread_id})
    await dispatch_hook("session.end", {"thread_id": thread_id})


def _build_non_interactive_header(
    assistant_id: str,
    thread_id: str,
    *,
    include_thread_link: bool = False,
) -> Text:
    """모델, 에이전트 및 스레드 정보를 사용하여 비대화형 모드 헤더를 빌드합니다.

    기본적으로 이 기능은 LangSmith 네트워크 조회를 방지하고 스레드 ID를 일반 텍스트로 렌더링합니다. 발신자는 하이퍼링크 확인을 선택할 수
    있습니다.

    Args:
        assistant_id: 에이전트 식별자.
        thread_id: 스레드 식별자.
        include_thread_link: 스레드 ID에 대한 LangSmith 링크를 확인하고 렌더링할지 여부입니다.

    Returns:
        서식이 지정된 헤더 줄이 있는 서식 있는 텍스트 개체입니다.

    """

    default_label = " (default)" if assistant_id == DEFAULT_AGENT_NAME else ""
    parts: list[tuple[str, str | Style]] = [
        (f"CLI: v{__version__}", "dim"),
        (" | ", "dim"),
        (f"Agent: {assistant_id}{default_label}", "dim"),
    ]

    if settings.model_name:
        parts.extend([(" | ", "dim"), (f"Model: {settings.model_name}", "dim")])

    parts.append((" | ", "dim"))

    thread_url = build_langsmith_thread_url(thread_id) if include_thread_link else None
    if thread_url:
        parts.extend(
            [
                ("Thread: ", "dim"),
                (thread_id, Style(dim=True, link=thread_url)),
            ]
        )
    else:
        parts.append((f"Thread: {thread_id}", "dim"))

    return Text.assemble(*parts)


async def run_non_interactive(
    message: str,
    assistant_id: str = "agent",
    model_name: str | None = None,
    model_params: dict[str, Any] | None = None,
    sandbox_type: str = "none",  # str (not None) to match argparse choices
    sandbox_id: str | None = None,
    sandbox_setup: str | None = None,
    *,
    profile_override: dict[str, Any] | None = None,
    quiet: bool = False,
    stream: bool = True,
    mcp_config_path: str | None = None,
    no_mcp: bool = False,
    trust_project_mcp: bool = False,
) -> int:
    """단일 작업을 비대화식으로 실행하고 종료합니다.

    에이전트는 자동 헤드리스 실행을 위한 시스템 프롬프트를 조정하는 `interactive=False`을 사용하여 생성됩니다(명확한 질문 없음, 합리적인
    가정).

    셸 액세스 및 자동 승인은 `--shell-allow-list`에 의해 제어됩니다.

    - 설정되지 않음 → 셸이 비활성화되고 다른 모든 도구는 자동 승인됩니다. - `recommended` 또는 명시적 목록 → 쉘 활성화, 명령은
    다음으로 제어됨
        허용 목록; 비쉘 도구는 무조건 승인됩니다.
    - `all` → 쉘 활성화, 모든 명령 허용, 모든 도구 자동 승인.

    참고: 시작 헤더 렌더링은 동기식 LangSmith URL 조회를 방지합니다. 백그라운드 스레드는 스레드 URL을 동시에 확인하며 가능한 경우 작업
    완료 후 결과가 표시됩니다.

    Args:
        message: 실행할 작업/메시지입니다.
        assistant_id: 메모리 저장을 위한 에이전트 식별자입니다.
        model_name: 사용할 선택적 모델 이름입니다.
        model_params: 모델에 전달할 `--model-params`의 추가 kwargs입니다.

            이는 구성 파일 값을 재정의합니다.
        sandbox_type: 샌드박스 유형(`'none'`, `'agentcore'`, `'daytona'`, `'langsmith'`,
                      `'modal'`, `'runloop'`).
        sandbox_id: 재사용할 선택적 기존 샌드박스 ID입니다.
        sandbox_setup: 생성 후 샌드박스에서 실행할 설정 스크립트의 선택적 경로입니다.
        profile_override: `--profile-override`의 추가 프로필 필드입니다.

            구성 파일 프로필 재정의 위에 병합됩니다.
        quiet: `True`인 경우 모든 콘솔 출력(헤더, 상태 메시지, 도구 알림, HITL 결정, 오류)이 stderr로 리디렉션되어 에이전트의
               응답 텍스트만 stdout에 표시됩니다.
        stream: `True`(기본값)이면 텍스트 청크가 도착하자마자 stdout에 기록됩니다.

            `False`인 경우 에이전트가 완료된 후 전체 응답이 버퍼링되어 한 번에 stdout에 기록됩니다.
        mcp_config_path: MCP 서버 JSON 구성 파일의 선택적 경로입니다. 자동 검색된 구성 위에 병합됩니다(가장 높은 우선순위).
        no_mcp: 모든 MCP 도구 로딩을 비활성화합니다.
        trust_project_mcp: `True`인 경우 프로젝트 수준 stdio MCP 서버를 허용합니다. `False`(기본값)이면 프로젝트
                           stdio 서버가 자동으로 건너뜁니다.

    Returns:
        Exit code: 0은 성공, 1은 오류, 130은 키보드 인터럽트입니다.

    """

    # stderr=True routes all console.print() to stderr; agent response text
    # uses _write_text() -> sys.stdout directly.
    console = Console(stderr=True) if quiet else Console()
    try:
        result = create_model(
            model_name,
            extra_kwargs=model_params,
            profile_overrides=profile_override,
        )
    except ModelConfigError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        return 1

    result.apply_to_settings()
    thread_id = generate_thread_id()

    from deepagents_cli.config import build_stream_config

    config: RunnableConfig = build_stream_config(
        thread_id, assistant_id, sandbox_type=sandbox_type
    )

    thread_url_lookup: ThreadUrlLookupState | None = None
    if not quiet:
        thread_url_lookup = _start_langsmith_thread_url_lookup(thread_id)
        console.print(Text("Running task non-interactively...", style="dim"))
        header = _build_non_interactive_header(assistant_id, thread_id)
        console.print(header)

    import asyncio

    from deepagents_cli.server_manager import server_session

    # Launch MCP preload concurrently with server startup
    mcp_task: asyncio.Task[Any] | None = None
    if not no_mcp and not quiet:
        try:
            from deepagents_cli.main import _preload_session_mcp_server_info

            mcp_task = asyncio.create_task(
                _preload_session_mcp_server_info(
                    mcp_config_path=mcp_config_path,
                    no_mcp=no_mcp,
                    trust_project_mcp=trust_project_mcp,
                )
            )
        except Exception:
            logger.warning("MCP metadata preload task creation failed", exc_info=True)

    try:
        enable_shell = bool(settings.shell_allow_list)
        shell_is_unrestricted = isinstance(
            settings.shell_allow_list, type(SHELL_ALLOW_ALL)
        )
        # Currently, non-shell tools have no HITL handler in non-interactive
        # mode, so interrupting on them just fragments LangSmith traces
        # without adding value. Gate only shell execution via middleware.
        use_auto_approve = not enable_shell or shell_is_unrestricted
        use_interrupt_shell_only = enable_shell and not shell_is_unrestricted
        # Extract the concrete allow-list to forward to the server subprocess.
        # settings.shell_allow_list is already validated at this point.
        restrictive_allow_list: list[str] | None = (
            list(settings.shell_allow_list)
            if use_interrupt_shell_only and settings.shell_allow_list
            else None
        )

        if not quiet:
            console.print(Text("Starting LangGraph server...", style="dim"))

        async with server_session(
            assistant_id=assistant_id,
            model_name=model_name,
            model_params=model_params,
            auto_approve=use_auto_approve,
            interrupt_shell_only=use_interrupt_shell_only,
            shell_allow_list=restrictive_allow_list,
            sandbox_type=sandbox_type,
            sandbox_id=sandbox_id,
            sandbox_setup=sandbox_setup,
            enable_shell=enable_shell,
            enable_ask_user=False,
            mcp_config_path=mcp_config_path,
            no_mcp=no_mcp,
            trust_project_mcp=trust_project_mcp,
            interactive=False,
        ) as (agent, _server_proc):
            # Collect MCP preload result (ran concurrently with server startup)
            if mcp_task is not None:
                try:
                    mcp_info = await mcp_task
                    if mcp_info:
                        tool_count = sum(len(s.tools) for s in mcp_info)
                        if tool_count:
                            label = "MCP tool" if tool_count == 1 else "MCP tools"
                            console.print(
                                f"[green]✓ Loaded {tool_count} {label}[/green]"
                            )
                except Exception:
                    logger.warning("MCP metadata preload failed", exc_info=True)

            if not quiet:
                console.print("[green]✓ Server ready[/green]")

            file_op_tracker = FileOpTracker(assistant_id=assistant_id, backend=None)

            await _run_agent_loop(
                agent,
                message,
                config,
                console,
                file_op_tracker,
                quiet=quiet,
                stream=stream,
                thread_url_lookup=thread_url_lookup,
            )

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
        return 130
    except HITLIterationLimitError as e:
        console.print(f"\n[red]{escape_markup(str(e))}[/red]")
        console.print(
            "[yellow]Hint: The agent may be repeatedly attempting commands "
            "that are not in the allow-list. Consider expanding the "
            "--shell-allow-list or adjusting the task.[/yellow]"
        )
        return 1
    except (ValueError, OSError) as e:
        logger.exception("Error during non-interactive execution")
        console.print(f"\n[red]Error: {escape_markup(str(e))}[/red]")
        return 1
    except Exception as e:
        logger.exception("Unexpected error during non-interactive execution")
        console.print(
            f"\n[red]Unexpected error ({type(e).__name__}): "
            f"{escape_markup(str(e))}[/red]"
        )
        return 1
    else:
        return 0
