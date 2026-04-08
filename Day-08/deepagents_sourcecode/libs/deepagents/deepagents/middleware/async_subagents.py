"""원격 Agent Protocol 서버에서 실행되는 비동기 서브에이전트 미들웨어 모듈.

비동기 서브에이전트는 LangGraph SDK를 사용하여 원격
Agent Protocol(https://github.com/langchain-ai/agent-protocol) 서버에서
백그라운드 실행(run)을 시작합니다.

동기 서브에이전트(완료까지 블로킹)와 달리, 비동기 서브에이전트는
즉시 task ID를 반환하여 메인 에이전트가 서브에이전트가 작업하는 동안
진행 상황을 모니터링하고 업데이트를 보낼 �� 있습니다.

LangGraph Platform(관리형) 및 자체 호스팅 서버와 호환됩니다.

## 핵심 개념

### 동기 vs 비동기 서브에이전트

| 항목 | SubAgentMiddleware (동기) | AsyncSubAgentMiddleware (비동기, 이 모듈) |
|------|--------------------------|------------------------------------------|
| 실행 | 블로킹 — 완료까지 대기 | 논블로킹 — 즉시 task_id 반환 |
| 위치 | 로컬 (같은 프로세스) | 원격 Agent Protocol 서버 |
| 통신 | 단방향 (결과만 반환) | 양방향 (업데이트 전송 가능) |
| 모니터링 | 불가 | check/list로 상태 조회 가능 |
| 취소 | 불가 | cancel로 실행 중 취소 가능 |

### 도구 세트 (5개)
1. `start_async_task` — 원격 서버에서 새 백그라운드 태스크 시작
2. `check_async_task` — 특정 태스크의 상태와 결과 조회
3. `update_async_task` — 실행 중인 태스크에 후속 지시 전송
4. `cancel_async_task` — 실행 중인 태스크 취소
5. `list_async_tasks` — 모든 추적 중인 태스크의 라이브 상태 조회

### 상태 관리
태스크 정보는 에이전트 상태의 `async_tasks` 딕셔너리에 저장됩니다.
`_tasks_reducer`를 통해 기존 태스크와 업데이트가 병합됩니다.
이를 통해 컨텍스트 압축/오프로딩 이후에도 태스크 정보가 유지됩니다.

### 클라이언트 캐싱
`_ClientCache` 클래스가 (url, headers) 쌍을 키로 하여 LangGraph SDK 클라이언트를
지연 생성하고 캐싱합니다. 동일 서버에 대한 중복 연결을 방지합니다.
"""

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Annotated, Any, Literal, NotRequired, TypedDict

from langchain.agents.middleware.types import AgentMiddleware, AgentState, ContextT, ModelRequest, ModelResponse, ResponseT
from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.types import Command
from langgraph_sdk import get_client, get_sync_client
from langgraph_sdk.client import LangGraphClient, SyncLangGraphClient
from langgraph_sdk.schema import Run
from pydantic import BaseModel, Field

from deepagents.middleware._utils import append_to_system_message

logger = logging.getLogger(__name__)


class AsyncSubAgent(TypedDict):
    """원격 Agent Protocol 서버에서 실행되는 비동기 서브에이전트 설정 스펙.

    비동기 서브에이전트는 LangGraph SDK를 통해 Agent Protocol 호환 서버에 연결합니다.
    메인 에이전트가 모니터링하고 업데이트할 수 있는 백그라운드 태스크로 실행됩니다.

    LangGraph Platform(관리형) 및 자체 호스팅 서버와 호환됩니다.
    LangGraph Platform의 인증은 SDK가 환경 변수(`LANGGRAPH_API_KEY`,
    `LANGSMITH_API_KEY`, 또는 `LANGCHAIN_API_KEY`)를 통해 자동으로 처리합니다.
    자체 호스팅 서버의 경우, `headers`를 통해 커스텀 인증을 전달합니다.
    """

    name: str
    """비동기 서브에이전트의 고유 식별자."""

    description: str
    """서브에이전트가 수행하는 작업 설명.
    메인 에이전트가 위임 판단 시 참고합니다."""

    graph_id: str
    """원격 서버의 그래프 이름 또는 어시스턴트 ID."""

    url: NotRequired[str]
    """Agent Protocol 서버의 URL.
    기본값은 LangGraph SDK의 기본 엔드포인트.
    로컬 서버에 ASGI 전송을 사용하려면 생략합니다."""

    headers: NotRequired[dict[str, str]]
    """원격 서버 요청에 포함할 추가 HTTP 헤더."""


class AsyncTask(TypedDict):
    """에이전트 상태에 저장되는 추적 대상 비동기 서브에이전트 태스크.

    각 태스크는 원격 서버의 thread와 run에 대한 참조를 포함하며,
    상태 변경 이력을 타임스탬프로 추적합니다.
    """

    task_id: str
    """태스크의 고유 식별자 (`thread_id`와 동일)."""

    agent_name: str
    """실행 중인 비동기 서브에이전트 유형의 이름."""

    thread_id: str
    """원격 서버의 스레드 ID."""

    run_id: str
    """해당 스레드에서 현재 실행 중인 run의 ID."""

    status: str
    """현재 태스크 상태 (예: `'running'`, `'success'`, `'error'`, `'cancelled'`).
    LangGraph SDK의 `Run.status`가 str 타입이므로 Literal 대신 str을 사용합니다.
    Literal을 사용하면 SDK 경계마다 `cast`가 필요해지기 때문입니다."""

    created_at: str
    """태스크 생성 시각의 ISO-8601 타임스탬프 (UTC, 초 단위).
    형식: `YYYY-MM-DDTHH:MM:SSZ` (예: `2024-01-15T10:30:00Z`)."""

    last_checked_at: str
    """SDK를 통해 태스크 상태를 마지막으로 확인한 시각의 ISO-8601 타임스탬프 (UTC).
    형식: `YYYY-MM-DDTHH:MM:SSZ`."""

    last_updated_at: str
    """태스크 상태가 변경되었거나, update 도구로 후속 메시지를 보낸 시각의 ISO-8601 타임스탬프.
    형식: `YYYY-MM-DDTHH:MM:SSZ`."""


def _tasks_reducer(
    existing: dict[str, AsyncTask] | None,
    update: dict[str, AsyncTask],
) -> dict[str, AsyncTask]:
    """태스크 업데이트를 기존 태스크 딕셔너리에 병합하는 리듀서.

    LangGraph의 Annotated 상태 관리에서 사용됩니다.
    기존 딕셔너리에 새 엔트리를 추가하거나, 동일 키의 엔트리를 업데이트합니다.

    Args:
        existing: 기존 태스크 딕셔너리. 초기화 시 None일 수 있음.
        update: 병합할 새 태스크 딕셔너리.

    Returns:
        병합된 태스크 딕셔너리.
    """
    merged = dict(existing or {})
    merged.update(update)
    return merged


class AsyncSubAgentState(AgentState):
    """비동기 서브에이전트 태스크 추적을 위한 상태 확장.

    에이전트 상태에 `async_tasks` 필드를 추가하여, 실행 중이거나
    완료된 비동기 태스크들의 정보를 저장합니다.
    `_tasks_reducer`를 통해 업데이트가 병합됩니다.
    """

    async_tasks: Annotated[NotRequired[dict[str, AsyncTask]], _tasks_reducer]


class StartAsyncTaskSchema(BaseModel):
    """start_async_task 도구의 입력 스키마."""

    description: str = Field(description="A detailed description of the task for the async subagent to perform.")
    subagent_type: str = Field(description="The type of async subagent to use. Must be one of the available types listed in the tool description.")


class CheckAsyncTaskSchema(BaseModel):
    """check_async_task 도구의 입력 스키마."""

    task_id: str = Field(description="The exact task_id string returned by start_async_task. Pass it verbatim.")


class UpdateAsyncTaskSchema(BaseModel):
    """update_async_task 도구의 입력 스키마."""

    task_id: str = Field(description="The exact task_id string returned by start_async_task. Pass it verbatim.")
    message: str = Field(description="Follow-up instructions or context to send to the subagent.")


class CancelAsyncTaskSchema(BaseModel):
    """cancel_async_task 도구의 입력 스키마."""

    task_id: str = Field(description="The exact task_id string returned by start_async_task. Pass it verbatim.")


class ListAsyncTasksSchema(BaseModel):
    """list_async_tasks 도구의 입력 스키마."""

    status_filter: Literal["running", "success", "error", "cancelled", "all"] | None = Field(
        default=None,
        description="Filter tasks by status. One of: 'running', 'success', 'error', 'cancelled', 'all'. Defaults to 'all'.",
    )


# start_async_task 도구의 설명 템플릿
# {available_agents}에 사용 가능한 비동기 에이전트 목록이 삽입됩니다.
ASYNC_TASK_TOOL_DESCRIPTION = """Start an async subagent on a remote server. The subagent runs in the background and returns a task ID immediately.

Available async agent types:
{available_agents}

## Usage notes:
1. This tool launches a background task and returns immediately with a task ID. Report the task ID to the user and stop — do NOT immediately check status.
2. Use `check_async_task` only when the user asks for a status update or result.
3. Use `update_async_task` to send new instructions to a running task.
4. Multiple async subagents can run concurrently — launch several and let them run in the background.
5. The subagent runs on a remote server, so it has its own tools and capabilities."""  # noqa: E501

# 시스템 프롬프트에 주입되는 비동기 서브에이전트 도구 사용 지침
# LLM에게 5개 도구의 워크플로우, 규칙, 사용 시점을 상세히 안내합니다.
ASYNC_TASK_SYSTEM_PROMPT = """## Async subagents (remote LangGraph servers)

You have access to async subagent tools that launch background tasks on remote LangGraph servers.

### Tools:
- `start_async_task`: Start a new background task. Returns a task ID immediately.
- `check_async_task`: Get current status and result of a task. Returns status + result (if complete).
- `update_async_task`: Send new instructions to a running task. Returns confirmation + updated status.
- `cancel_async_task`: Stop a running task. Returns confirmation.
- `list_async_tasks`: List all tracked tasks with live statuses. Returns summary of all tasks.

### Workflow:
1. **Start** — Use `start_async_task` to start a task. Report the task ID to the user and stop.
   Do NOT immediately check the status — the task runs in the background while you and the user continue other work.
2. **Check (on request)** — Only use `check_async_task` when the user explicitly asks for a status update or
   result. If the status is "running", report that and stop — do not poll in a loop.
3. **Update** (optional) — Use `update_async_task` to send new instructions to a running task. This interrupts
   the current run and starts a fresh one on the same thread. The task_id stays the same.
4. **Cancel** (optional) — Use `cancel_async_task` to stop a task that is no longer needed.
5. **Collect** — When `check_async_task` returns status "success", the result is included in the response.
6. **List** — Use `list_async_tasks` to see live statuses for all tasks at once, or to recall task IDs after context compaction.

### Critical rules:
- After launching, ALWAYS return control to the user immediately. Never auto-check after launching.
- Never poll `check_async_task` in a loop. Check once per user request, then stop.
- If a check returns "running", tell the user and wait for them to ask again.
- Task statuses in conversation history are ALWAYS stale — a task that was "running" may now be done.
  NEVER report a status from a previous tool result. ALWAYS call a tool to get the current status:
  use `list_async_tasks` when the user asks about multiple tasks or "all tasks",
  use `check_async_task` when the user asks about a specific task.
- Always show the full task_id — never truncate or abbreviate it.

### When to use async subagents:
- Long-running tasks that would block the main agent
- Tasks that benefit from running on specialized remote deployments
- When you want to run multiple tasks concurrently and collect results later"""


def _resolve_headers(spec: AsyncSubAgent) -> dict[str, str]:
    """원격 Agent Protocol 서버용 HTTP 헤더를 구성합니다.

    기본적으로 `x-auth-scheme: langsmith`를 추가합니다 (이미 제공된 경우 제외).
    이 헤더가 필요 없는 자체 호스팅 서버에서는 일반적으로 무시됩니다.
    `AsyncSubAgent` 설정의 `headers` 필드로 오버라이드할 수 있습니다.

    Args:
        spec: 비동기 서브에이전트 설정 스펙.

    Returns:
        해석된 헤더 딕셔너리.
    """
    headers: dict[str, str] = dict(spec.get("headers") or {})
    if "x-auth-scheme" not in headers:
        headers["x-auth-scheme"] = "langsmith"
    return headers


class _ClientCache:
    """(url, headers) 쌍을 키로 하여 Agent Protocol 클라이언트를 지연 생성하고 캐싱하는 클래스.

    동일 서버에 대한 중복 클라이언트 생성을 방지합니다.
    동기(SyncLangGraphClient)와 비동기(LangGraphClient) 클라이언트를 각각 관리합니다.

    캐시 키 구성:
        (url 또는 None, frozenset(headers.items()))
        — url과 headers가 모두 동일하면 같은 클라이언트를 재사용
    """

    def __init__(self, agents: dict[str, AsyncSubAgent]) -> None:
        """클라이언트 캐시를 초기화합니다.

        Args:
            agents: 이름 → 서브에이전트 스펙 매핑 딕셔너리.
        """
        self._agents = agents
        # 동기 클라이언트 캐시: (url, frozenset(headers)) → SyncLangGraphClient
        self._sync: dict[tuple[str | None, frozenset[tuple[str, str]]], SyncLangGraphClient] = {}
        # 비동기 클라이언트 캐시: (url, frozenset(headers)) → LangGraphClient
        self._async: dict[tuple[str | None, frozenset[tuple[str, str]]], LangGraphClient] = {}

    def _cache_key(self, spec: AsyncSubAgent) -> tuple[str | None, frozenset[tuple[str, str]]]:
        """에이전트 스펙의 url과 해석된 headers로 캐시 키를 생성합니다.

        Args:
            spec: 비동기 서브에이전트 설정 스펙.

        Returns:
            (url 또는 None, frozenset(headers.items())) 튜플.
        """
        return (spec.get("url"), frozenset(_resolve_headers(spec).items()))

    def get_sync(self, name: str) -> SyncLangGraphClient:
        """지정된 에이전트의 동기 클라이언트를 가져오거나 생성합니다.

        ASGI 전송(url=None)은 동기 호출을 지원하지 않으므로,
        url이 없는 에이전트에 대해서는 ValueError를 발생시킵니다.

        Args:
            name: 에이전트 이름.

        Returns:
            동기 LangGraph 클라이언트.

        Raises:
            ValueError: url이 설정되지 않은 에이전트에 동기 클라이언트를 요청한 경우.
        """
        spec = self._agents[name]
        if spec.get("url") is None:
            msg = f"Async subagent '{name}' has no url configured. ASGI transport (url=None) requires async invocation."
            raise ValueError(msg)
        key = self._cache_key(spec)
        if key not in self._sync:
            self._sync[key] = get_sync_client(
                url=spec.get("url"),
                headers=_resolve_headers(spec),
            )
        return self._sync[key]

    def get_async(self, name: str) -> LangGraphClient:
        """지정된 에이전트의 비동기 클라이언트를 가져오거나 생성합니다.

        url=None인 경우 ASGI 전송(로컬 서버)을 사용합니다.

        Args:
            name: 에이전트 이름.

        Returns:
            비동기 LangGraph 클라이언트.
        """
        spec = self._agents[name]
        key = self._cache_key(spec)
        if key not in self._async:
            self._async[key] = get_client(
                url=spec.get("url"),
                headers=_resolve_headers(spec),
            )
        return self._async[key]


def _validate_agent_type(agent_map: dict[str, AsyncSubAgent], agent_type: str) -> str | None:
    """에이전트 유형이 유효한지 검증합니다.

    Args:
        agent_map: 이름 → 서브에이전트 스펙 매핑.
        agent_type: 검증할 에이전트 유형 이름.

    Returns:
        유효하지 않으면 오류 메시지 문자열, 유효하면 None.
    """
    if agent_type not in agent_map:
        allowed = ", ".join(f"`{k}`" for k in agent_map)
        return f"Unknown async subagent type `{agent_type}`. Available types: {allowed}"
    return None


def _build_start_tool(
    agent_map: dict[str, AsyncSubAgent],
    clients: _ClientCache,
    tool_description: str,
) -> StructuredTool:
    """start_async_task 도구를 빌드합니다.

    원격 서버에 새 스레드를 생성하고, 해당 스레드에서 run을 시작한 후,
    task_id(= thread_id)를 반환합니다.

    태스크 정보는 Command를 통해 에이전트 상태의 async_tasks에 저장됩니다.

    Args:
        agent_map: 이름 → 서브에이전트 스펙 매핑.
        clients: 클라이언트 캐시 인스턴스.
        tool_description: 도구 설명 문자열.

    Returns:
        start_async_task StructuredTool.
    """

    def start_async_task(
        description: str,
        subagent_type: str,
        runtime: ToolRuntime,
    ) -> str | Command:
        """비동기 태스크를 시작합니다 (동기 버전).

        원격 서버에 스레드를 생성하고 run을 시작합니다.
        즉시 task_id를 반환하며, 태스크는 백그라운드에서 실행됩니다.

        Args:
            description: 서브에이전트가 수행할 작업의 상세 설명.
            subagent_type: 사용할 비동기 서브에이전트 유형.
            runtime: 도구 런타임.

        Returns:
            성공 시 Command (태스크 상태 업데이트 + ToolMessage),
            실패 시 오류 메시지 문자열.
        """
        error = _validate_agent_type(agent_map, subagent_type)
        if error:
            return error
        spec = agent_map[subagent_type]
        try:
            client = clients.get_sync(subagent_type)
            # 새 스레드 생성 후, 해당 스레드에서 run 시작
            thread = client.threads.create()
            run = client.runs.create(
                thread_id=thread["thread_id"],
                assistant_id=spec["graph_id"],
                input={"messages": [{"role": "user", "content": description}]},
            )
        except Exception as e:  # noqa: BLE001  # LangGraph SDK는 타입이 지정되지 않은 예외를 발생시킴
            logger.warning("Failed to launch async subagent '%s': %s", subagent_type, e)
            return f"Failed to launch async subagent '{subagent_type}': {e}"

        # task_id로 thread_id를 사용 (1:1 매핑)
        task_id = thread["thread_id"]
        now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        task: AsyncTask = {
            "task_id": task_id,
            "agent_name": subagent_type,
            "thread_id": task_id,
            "run_id": run["run_id"],
            "status": "running",
            "created_at": now,
            "last_checked_at": now,
            "last_updated_at": now,
        }
        msg = f"Launched async subagent. task_id: {task_id}"
        # Command를 통해 태스크 정보를 에이전트 상태에 저장
        return Command(
            update={
                "messages": [ToolMessage(msg, tool_call_id=runtime.tool_call_id)],
                "async_tasks": {task_id: task},
            }
        )

    async def astart_async_task(
        description: str,
        subagent_type: str,
        runtime: ToolRuntime,
    ) -> str | Command:
        """비동기 태스크를 시작합니다 (비동기 버전).

        동기 버전과 동일한 로직이지만, 비동기 클라이언트를 사용합니다.

        Args:
            description: 서브에이전트가 수행할 작업의 상세 설명.
            subagent_type: 사용할 비동기 서브에이전트 유형.
            runtime: 도구 런타임.

        Returns:
            성공 시 Command, 실패 시 오류 메시지 문자열.
        """
        error = _validate_agent_type(agent_map, subagent_type)
        if error:
            return error
        spec = agent_map[subagent_type]
        try:
            client = clients.get_async(subagent_type)
            thread = await client.threads.create()
            run = await client.runs.create(
                thread_id=thread["thread_id"],
                assistant_id=spec["graph_id"],
                input={"messages": [{"role": "user", "content": description}]},
            )
        except Exception as e:  # noqa: BLE001  # LangGraph SDK는 타입이 지정되지 않은 예외를 발생시킴
            logger.warning("Failed to launch async subagent '%s': %s", subagent_type, e)
            return f"Failed to launch async subagent '{subagent_type}': {e}"
        task_id = thread["thread_id"]
        now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        task: AsyncTask = {
            "task_id": task_id,
            "agent_name": subagent_type,
            "thread_id": task_id,
            "run_id": run["run_id"],
            "status": "running",
            "created_at": now,
            "last_checked_at": now,
            "last_updated_at": now,
        }
        msg = f"Launched async subagent. task_id: {task_id}"
        return Command(
            update={
                "messages": [ToolMessage(msg, tool_call_id=runtime.tool_call_id)],
                "async_tasks": {task_id: task},
            }
        )

    return StructuredTool.from_function(
        name="start_async_task",
        func=start_async_task,
        coroutine=astart_async_task,
        description=tool_description,
        infer_schema=False,
        args_schema=StartAsyncTaskSchema,
    )


def _build_check_result(
    run: Run,
    thread_id: str,
    thread_values: dict[str, Any],
) -> dict[str, Any]:
    """run의 현재 상태와 스레드 값에서 확인 결과 딕셔너리를 구성합니다.

    상태에 따라 다른 정보를 포함합니다:
    - success: 스레드의 마지막 메시지 내용을 결과로 포함
    - error: 오류 상세 정보 포함
    - 기타: 상태와 thread_id만 포함

    Args:
        run: LangGraph SDK의 Run 객체.
        thread_id: 원격 서버의 스레드 ID.
        thread_values: 스레드의 현재 상태값 (성공 시 메시지 포함).

    Returns:
        status, thread_id, 그리고 선택적으로 result 또는 error를 포함하는 딕셔너리.
    """
    result: dict[str, Any] = {
        "status": run["status"],
        "thread_id": thread_id,
    }
    if run["status"] == "success":
        # 성공 시 스레드의 마지막 메시지를 결과로 추출
        messages = thread_values.get("messages", []) if isinstance(thread_values, dict) else []
        if messages:
            last = messages[-1]
            result["result"] = last.get("content", "") if isinstance(last, dict) else str(last)
        else:
            result["result"] = "(completed with no output messages)"
    elif run["status"] == "error":
        # 오류 시 오류 상세 정보 추출
        error_detail = run.get("error")
        result["error"] = str(error_detail) if error_detail else "The async subagent encountered an error."
    return result


def _build_check_command(
    result: dict[str, Any],
    task: AsyncTask,
    tool_call_id: str | None,
) -> Command:
    """확인 결과를 위한 Command 업데이트를 구성합니다.

    확인 시점의 타임스탬프를 업데이트하고, 상태가 변경되었으면
    last_updated_at도 함께 업데이트합니다.

    Args:
        result: _build_check_result에서 생성된 결과 딕셔너리.
        task: 기존 태스크 정보.
        tool_call_id: 원본 도구 호출 ID.

    Returns:
        태스크 상태 업데이트와 결과 ToolMessage가 포함된 Command.
    """
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    # 상태가 변경되었으면 last_updated_at도 갱신, 아니면 이전 값 유지
    last_updated_at = now if task["status"] != result["status"] else task["last_updated_at"]
    updated_task = AsyncTask(
        task_id=task["task_id"],
        agent_name=task["agent_name"],
        thread_id=task["thread_id"],
        run_id=task["run_id"],
        status=result["status"],
        created_at=task["created_at"],
        last_checked_at=now,
        last_updated_at=last_updated_at,
    )
    return Command(
        update={
            "messages": [ToolMessage(json.dumps(result), tool_call_id=tool_call_id)],
            "async_tasks": {task["task_id"]: updated_task},
        }
    )


def _resolve_tracked_task(
    task_id: str,
    runtime: ToolRuntime,
) -> AsyncTask | str:
    """상태에서 task_id로 추적 중인 태스크를 조회합니다.

    check, update, cancel 도구에서 공통으로 사용됩니다.

    Args:
        task_id: 조회할 태스크 ID (thread_id).
        runtime: 도구 런타임 (상태 접근용).

    Returns:
        성공 시 추적 중인 AsyncTask, 실패 시 오류 메시지 문자열.
    """
    tasks: dict[str, AsyncTask] = runtime.state.get("async_tasks") or {}
    tracked = tasks.get(task_id.strip())
    if not tracked:
        return f"No tracked task found for task_id: {task_id!r}"
    return tracked


def _build_check_tool(  # noqa: C901  # 필수적인 오류 처리로 인한 복잡도
    clients: _ClientCache,
) -> StructuredTool:
    """check_async_task 도구를 빌드합니다.

    태스크의 현재 상태를 원격 서버에서 조회하고, 성공 시 결과를 포함합니다.

    Args:
        clients: 클라이언트 캐시 인스턴스.

    Returns:
        check_async_task StructuredTool.
    """

    def check_async_task(
        task_id: str,
        runtime: ToolRuntime,
    ) -> str | Command:
        """비동기 태스크의 상태를 확인합니다 (동기 버전).

        원격 서버에서 run 상태를 조회하고, 성공 시 스레드 값에서 결과를 추출합니다.

        Args:
            task_id: 확인할 태스크 ID.
            runtime: 도구 런타임.

        Returns:
            성공 시 Command (상태 업데이트 + 결과 ToolMessage),
            실패 시 오류 메시지 문자열.
        """
        task = _resolve_tracked_task(task_id, runtime)
        if isinstance(task, str):
            return task

        client = clients.get_sync(task["agent_name"])
        try:
            run = client.runs.get(thread_id=task["thread_id"], run_id=task["run_id"])
        except Exception as e:  # noqa: BLE001  # LangGraph SDK 예외
            return f"Failed to get run status: {e}"

        # 성공 상태인 경우에만 스레드 값을 가져옴 (결과 메시지 추출을 위해)
        thread_values: dict[str, Any] = {}
        if run["status"] == "success":
            try:
                thread = client.threads.get(thread_id=task["thread_id"])
                thread_values = thread.get("values") or {}
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to fetch thread values for task %s: %s", task["task_id"], e)

        result = _build_check_result(run, task["thread_id"], thread_values)
        return _build_check_command(result, task, runtime.tool_call_id)

    async def acheck_async_task(
        task_id: str,
        runtime: ToolRuntime,
    ) -> str | Command:
        """비동기 태스크의 상태를 확인합니다 (비동기 버전).

        Args:
            task_id: 확인할 태스크 ID.
            runtime: 도구 런타임.

        Returns:
            성공 시 Command, 실패 시 오류 메시지 문자열.
        """
        task = _resolve_tracked_task(task_id, runtime)
        if isinstance(task, str):
            return task

        client = clients.get_async(task["agent_name"])
        try:
            run = await client.runs.get(thread_id=task["thread_id"], run_id=task["run_id"])
        except Exception as e:  # noqa: BLE001
            return f"Failed to get run status: {e}"

        thread_values: dict[str, Any] = {}
        if run["status"] == "success":
            try:
                thread = await client.threads.get(thread_id=task["thread_id"])
                thread_values = thread.get("values") or {}
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to fetch thread values for task %s: %s", task["task_id"], e)

        result = _build_check_result(run, task["thread_id"], thread_values)
        return _build_check_command(result, task, runtime.tool_call_id)

    return StructuredTool.from_function(
        name="check_async_task",
        func=check_async_task,
        coroutine=acheck_async_task,
        description="Check the status of an async subagent task. Returns the current status and, if complete, the result.",
        infer_schema=False,
        args_schema=CheckAsyncTaskSchema,
    )


def _build_update_tool(
    agent_map: dict[str, AsyncSubAgent],
    clients: _ClientCache,
) -> StructuredTool:
    """update_async_task 도구를 빌드합니다.

    동일 스레드에 새 run을 생성하여 후속 메시지를 전송합니다.
    서브에이전트는 전체 대화 히스토리(원래 태스크 + 이전 결과)와
    새 메시지를 함께 봅니다. task_id는 동일하게 유지되고,
    내부 run_id만 업데이트됩니다.

    Args:
        agent_map: 이름 → 서브에이전트 스펙 매핑.
        clients: 클라이언트 캐시 인스턴스.

    Returns:
        update_async_task StructuredTool.
    """

    def update_async_task(
        task_id: str,
        message: str,
        runtime: ToolRuntime,
    ) -> str | Command:
        """실행 중인 비동기 태스크에 후속 지시를 전송합니다 (동기 버전).

        현재 run을 중단(interrupt)하고 동일 스레드에서 새 run을 시작합니다.
        `multitask_strategy="interrupt"`를 사용하여 기존 실행을 안전하게 중단합니다.

        Args:
            task_id: 업데이트할 태스크 ID.
            message: 서브에이전트에게 보낼 후속 지시 또는 컨텍스트.
            runtime: 도구 런타임.

        Returns:
            성공 시 Command (업데이트된 태스크 정보 + ToolMessage),
            실패 시 오류 메시지 문자열.
        """
        tracked = _resolve_tracked_task(task_id, runtime)
        if isinstance(tracked, str):
            return tracked
        spec = agent_map[tracked["agent_name"]]
        try:
            client = clients.get_sync(tracked["agent_name"])
            # 동일 스레드에 새 run 생성 (interrupt로 기존 run 중단)
            run = client.runs.create(
                thread_id=tracked["thread_id"],
                assistant_id=spec["graph_id"],
                input={"messages": [{"role": "user", "content": message}]},
                multitask_strategy="interrupt",
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to update async subagent '%s': %s", tracked["agent_name"], e)
            return f"Failed to update async subagent: {e}"
        now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        task: AsyncTask = {
            "task_id": tracked["task_id"],
            "agent_name": tracked["agent_name"],
            "thread_id": tracked["thread_id"],
            "run_id": run["run_id"],  # 새 run_id로 업데이트
            "status": "running",       # 새 run이므로 다시 running 상태
            "created_at": tracked["created_at"],
            "last_checked_at": tracked["last_checked_at"],
            "last_updated_at": now,
        }
        msg = f"Updated async subagent. task_id: {tracked['task_id']}"
        return Command(
            update={
                "messages": [ToolMessage(msg, tool_call_id=runtime.tool_call_id)],
                "async_tasks": {tracked["task_id"]: task},
            }
        )

    async def aupdate_async_task(
        task_id: str,
        message: str,
        runtime: ToolRuntime,
    ) -> str | Command:
        """실행 중인 비동기 태스크에 후속 지시를 전송합니다 (비동기 버전).

        동기 버전과 동일한 로직이지만, 비동기 클라이언트를 사용합니다.

        Args:
            task_id: 업데이트할 태스크 ID.
            message: 서브에이전트에게 보낼 후속 지시.
            runtime: 도구 런타임.

        Returns:
            성공 시 Command, 실패 시 오류 메시지 문자열.
        """
        tracked = _resolve_tracked_task(task_id, runtime)
        if isinstance(tracked, str):
            return tracked
        spec = agent_map[tracked["agent_name"]]
        try:
            client = clients.get_async(tracked["agent_name"])
            run = await client.runs.create(
                thread_id=tracked["thread_id"],
                assistant_id=spec["graph_id"],
                input={"messages": [{"role": "user", "content": message}]},
                multitask_strategy="interrupt",
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to update async subagent '%s': %s", tracked["agent_name"], e)
            return f"Failed to update async subagent: {e}"
        now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        task: AsyncTask = {
            "task_id": tracked["task_id"],
            "agent_name": tracked["agent_name"],
            "thread_id": tracked["thread_id"],
            "run_id": run["run_id"],
            "status": "running",
            "created_at": tracked["created_at"],
            "last_checked_at": tracked["last_checked_at"],
            "last_updated_at": now,
        }
        msg = f"Updated async subagent. task_id: {tracked['task_id']}"
        return Command(
            update={
                "messages": [ToolMessage(msg, tool_call_id=runtime.tool_call_id)],
                "async_tasks": {tracked["task_id"]: task},
            }
        )

    return StructuredTool.from_function(
        name="update_async_task",
        func=update_async_task,
        coroutine=aupdate_async_task,
        description=(
            "Send updated instructions to an async subagent. Interrupts the current run and starts "
            "a new one on the same thread, so the subagent sees the full conversation history plus "
            "your new message. The task_id remains the same."
        ),
        infer_schema=False,
        args_schema=UpdateAsyncTaskSchema,
    )


def _build_cancel_tool(
    clients: _ClientCache,
) -> StructuredTool:
    """cancel_async_task 도구를 빌드합니다.

    실행 중인 태스크의 run을 취소하고, 태스크 상태를 'cancelled'로 업데이트합니다.

    Args:
        clients: 클라이언트 캐시 인스턴스.

    Returns:
        cancel_async_task StructuredTool.
    """

    def cancel_async_task(
        task_id: str,
        runtime: ToolRuntime,
    ) -> str | Command:
        """실행 중인 비동기 태스크를 취소합니다 (동기 버전).

        Args:
            task_id: 취소할 태스크 ID.
            runtime: 도구 런타임.

        Returns:
            성공 시 Command (cancelled 상태 + ToolMessage),
            실패 시 오류 메시지 문자열.
        """
        tracked = _resolve_tracked_task(task_id, runtime)
        if isinstance(tracked, str):
            return tracked

        client = clients.get_sync(tracked["agent_name"])
        try:
            client.runs.cancel(thread_id=tracked["thread_id"], run_id=tracked["run_id"])
        except Exception as e:  # noqa: BLE001
            return f"Failed to cancel run: {e}"
        now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        updated = AsyncTask(
            task_id=tracked["task_id"],
            agent_name=tracked["agent_name"],
            thread_id=tracked["thread_id"],
            run_id=tracked["run_id"],
            status="cancelled",
            created_at=tracked["created_at"],
            last_checked_at=now,
            last_updated_at=now,
        )
        msg = f"Cancelled async subagent task: {tracked['task_id']}"
        return Command(
            update={
                "messages": [ToolMessage(msg, tool_call_id=runtime.tool_call_id)],
                "async_tasks": {tracked["task_id"]: updated},
            }
        )

    async def acancel_async_task(
        task_id: str,
        runtime: ToolRuntime,
    ) -> str | Command:
        """실행 중인 비동기 태스크를 취소합니다 (비동기 버전).

        Args:
            task_id: 취소할 태스크 ID.
            runtime: 도구 런타임.

        Returns:
            성공 시 Command, 실패 시 오류 메시지 문자열.
        """
        tracked = _resolve_tracked_task(task_id, runtime)
        if isinstance(tracked, str):
            return tracked

        client = clients.get_async(tracked["agent_name"])
        try:
            await client.runs.cancel(thread_id=tracked["thread_id"], run_id=tracked["run_id"])
        except Exception as e:  # noqa: BLE001
            return f"Failed to cancel run: {e}"
        now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        updated = AsyncTask(
            task_id=tracked["task_id"],
            agent_name=tracked["agent_name"],
            thread_id=tracked["thread_id"],
            run_id=tracked["run_id"],
            status="cancelled",
            created_at=tracked["created_at"],
            last_checked_at=now,
            last_updated_at=now,
        )
        msg = f"Cancelled async subagent task: {tracked['task_id']}"
        return Command(
            update={
                "messages": [ToolMessage(msg, tool_call_id=runtime.tool_call_id)],
                "async_tasks": {tracked["task_id"]: updated},
            }
        )

    return StructuredTool.from_function(
        name="cancel_async_task",
        func=cancel_async_task,
        coroutine=acancel_async_task,
        description="Cancel a running async subagent task. Use this to stop a task that is no longer needed.",
        infer_schema=False,
        args_schema=CancelAsyncTaskSchema,
    )


# 더 이상 변경되지 않는 종료 상태 집합
# 이 상태의 태스크는 라이브 상태 조회를 건너뛸 수 있어 불필요한 API 호출을 방지합니다.
_TERMINAL_STATUSES = frozenset({"cancelled", "success", "error", "timeout", "interrupted"})
"""더 이상 변경되지 않는 태스크 종료 상태 집합. 라이브 상태 조회를 건너뛸 수 있습니다."""


def _fetch_live_status(clients: _ClientCache, task: AsyncTask) -> str:
    """서버에서 현재 run 상태를 조회합니다 (동기 버전).

    종료 상태인 태스크는 서버 조회 없이 캐시된 상태를 반환합니다.
    오류 발생 시에도 캐시된 상태를 반환하여 장애 전파를 방지합니다.

    Args:
        clients: 클라이언트 캐시.
        task: 상태를 조회할 태스크.

    Returns:
        현재 run 상태 문자열.
    """
    # 종료 상태는 서버 조회 불필요
    if task["status"] in _TERMINAL_STATUSES:
        return task["status"]
    try:
        client = clients.get_sync(task["agent_name"])
        run = client.runs.get(thread_id=task["thread_id"], run_id=task["run_id"])
        return run["status"]
    except Exception:  # noqa: BLE001
        logger.warning(
            "Failed to fetch live status for task %s (agent=%s), returning cached status %r",
            task["task_id"],
            task["agent_name"],
            task["status"],
            exc_info=True,
        )
        return task["status"]


async def _afetch_live_status(clients: _ClientCache, task: AsyncTask) -> str:
    """서버에서 현재 run 상태를 조회합니다 (비동기 버전).

    동기 버전 `_fetch_live_status`와 동일한 로직이지만, 비동기 클라이언트를 사용합니다.

    Args:
        clients: 클라이언트 캐시.
        task: 상태를 조회할 태스크.

    Returns:
        현재 run 상태 문자열.
    """
    if task["status"] in _TERMINAL_STATUSES:
        return task["status"]
    try:
        client = clients.get_async(task["agent_name"])
        run = await client.runs.get(thread_id=task["thread_id"], run_id=task["run_id"])
        return run["status"]
    except Exception:  # noqa: BLE001
        logger.warning(
            "Failed to fetch live status for task %s (agent=%s), returning cached status %r",
            task["task_id"],
            task["agent_name"],
            task["status"],
            exc_info=True,
        )
        return task["status"]


def _format_task_entry(task: AsyncTask, status: str) -> str:
    """단일 태스크를 목록 출력용 표시 문자열로 포맷합니다.

    Args:
        task: 포맷할 태스크.
        status: 표시할 상태 (라이브 상태일 수 있음).

    Returns:
        "- task_id: ... agent: ... status: ..." 형식의 문자열.
    """
    return f"- task_id: {task['task_id']}  agent: {task['agent_name']}  status: {status}"


def _filter_tasks(
    tasks: dict[str, AsyncTask],
    status_filter: str | None,
) -> list[AsyncTask]:
    """에이전트 상태의 캐시된 상태를 기준으로 태스크를 필터링합니다.

    필터링은 캐시된 상태에 대해 수행되며, 라이브 서버 상태가 아닙니다.
    라이브 상태는 필터링 후 호출하는 도구에서 조회됩니다.

    Args:
        tasks: 상태에 저장된 모든 추적 중인 태스크.
        status_filter: None 또는 'all'이면 모든 태스크 반환.
            그 외에는 캐시된 상태가 일치하는 태스크만 반환.

    Returns:
        필터링된 태스크 리스트.
    """
    if not status_filter or status_filter == "all":
        return list(tasks.values())
    return [task for task in tasks.values() if task["status"] == status_filter]


def _build_list_tasks_tool(clients: _ClientCache) -> StructuredTool:
    """list_async_tasks 도구를 빌드합니다.

    모든 추적 중인 태스크의 라이브 상태를 조회하고, 포맷된 목록을 반환합니다.
    비동기 버전에서는 asyncio.gather를 사용하여 모든 태스크의 상태를 병렬로 조회합니다.

    Args:
        clients: 클라이언트 캐시 인스턴스.

    Returns:
        list_async_tasks StructuredTool.
    """

    def list_async_tasks(
        runtime: ToolRuntime,
        status_filter: Literal["running", "success", "error", "cancelled", "all"] | None = None,
    ) -> str | Command:
        """추적 중인 모든 비동기 태스크를 라이브 상태와 함께 나열합니다 (동기 버전).

        Args:
            runtime: 도구 런타임 (상태 접근용).
            status_filter: 상태별 필터. None 또는 'all'이면 전체 조회.

        Returns:
            태스크가 없으면 안내 문자열,
            있으면 Command (업데이트된 태스크 상태 + 목록 ToolMessage).
        """
        tasks: dict[str, AsyncTask] = runtime.state.get("async_tasks") or {}
        filtered = _filter_tasks(tasks, status_filter)
        if not filtered:
            return "No async subagent tasks tracked."

        updated_tasks: dict[str, AsyncTask] = {}
        entries: list[str] = []
        now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

        # 각 태스크의 라이브 상태를 순차적으로 조회
        for task in filtered:
            status = _fetch_live_status(clients, task)
            entries.append(_format_task_entry(task, status))
            last_updated_at = now if status != task["status"] else task["last_updated_at"]
            updated_tasks[task["task_id"]] = AsyncTask(
                task_id=task["task_id"],
                agent_name=task["agent_name"],
                thread_id=task["thread_id"],
                run_id=task["run_id"],
                status=status,
                created_at=task["created_at"],
                last_checked_at=now,
                last_updated_at=last_updated_at,
            )
        msg = f"{len(entries)} tracked task(s):\n" + "\n".join(entries)
        return Command(
            update={
                "messages": [ToolMessage(msg, tool_call_id=runtime.tool_call_id)],
                "async_tasks": updated_tasks,
            }
        )

    async def alist_async_tasks(
        runtime: ToolRuntime,
        status_filter: Literal["running", "success", "error", "cancelled", "all"] | None = None,
    ) -> str | Command:
        """추적 중인 모든 비동기 태스크를 라이브 상태와 함께 나열합니다 (비동기 버전).

        asyncio.gather를 사용하여 모든 태스크의 라이브 상태를 병렬로 조회합니다.

        Args:
            runtime: 도구 런타임.
            status_filter: 상태별 필터.

        Returns:
            태스크가 없으면 안내 문자열, 있으면 Command.
        """
        tasks: dict[str, AsyncTask] = runtime.state.get("async_tasks") or {}
        filtered = _filter_tasks(tasks, status_filter)
        if not filtered:
            return "No async subagent tasks tracked."

        # 모든 태스크의 라이브 상태를 병렬로 조회 (비동기의 핵심 장점)
        statuses = await asyncio.gather(*(_afetch_live_status(clients, task) for task in filtered))

        updated_tasks: dict[str, AsyncTask] = {}
        entries: list[str] = []
        now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

        for task, status in zip(filtered, statuses, strict=True):
            entries.append(_format_task_entry(task, status))
            last_updated_at = now if status != task["status"] else task["last_updated_at"]
            updated_tasks[task["task_id"]] = AsyncTask(
                task_id=task["task_id"],
                agent_name=task["agent_name"],
                thread_id=task["thread_id"],
                run_id=task["run_id"],
                status=status,
                created_at=task["created_at"],
                last_checked_at=now,
                last_updated_at=last_updated_at,
            )
        msg = f"{len(entries)} tracked task(s):\n" + "\n".join(entries)
        return Command(
            update={
                "messages": [ToolMessage(msg, tool_call_id=runtime.tool_call_id)],
                "async_tasks": updated_tasks,
            }
        )

    return StructuredTool.from_function(
        name="list_async_tasks",
        func=list_async_tasks,
        coroutine=alist_async_tasks,
        description=(
            "List tracked async subagent tasks with their current live statuses. "
            "By default shows all tasks. Use `status_filter` to narrow by status "
            "(e.g. 'running', 'success', 'error', 'cancelled'). "
            "Use `check_async_task` to get the full result of a specific completed task."
        ),
        infer_schema=False,
        args_schema=ListAsyncTasksSchema,
    )


def _build_async_subagent_tools(
    agents: list[AsyncSubAgent],
) -> list[StructuredTool]:
    """에이전트 스펙으로부터 비동기 서브에이전트 도구 세트를 빌드합니다.

    5개의 도구(start, check, update, cancel, list)를 생성합니다.
    내부적으로 에이전트 맵과 클라이언트 캐시를 구성하여 각 도구에 전달합니다.

    Args:
        agents: 비동기 서브에이전트 설정 스펙 리스트.

    Returns:
        launch, check, update, cancel, list 작업을 위한 StructuredTool 리스트.
    """
    # 이름 → 스펙 매핑 딕셔너리 생성
    agent_map: dict[str, AsyncSubAgent] = {a["name"]: a for a in agents}

    # 클라이언트 캐시 생성 (url+headers 기반 지연 생성 및 재사용)
    clients = _ClientCache(agent_map)

    # start 도구의 설명에 사용 가능한 에이전트 목록 삽입
    agents_desc = "\n".join(f"- {a['name']}: {a['description']}" for a in agents)
    launch_desc = ASYNC_TASK_TOOL_DESCRIPTION.format(available_agents=agents_desc)

    return [
        _build_start_tool(agent_map, clients, launch_desc),
        _build_check_tool(clients),
        _build_update_tool(agent_map, clients),
        _build_cancel_tool(clients),
        _build_list_tasks_tool(clients),
    ]


class AsyncSubAgentMiddleware(AgentMiddleware[Any, ContextT, ResponseT]):
    """원격 Agent Protocol 서버의 비동기 서브에이전트를 위한 미들웨어.

    이 미들웨어는 원격 Agent Protocol 서버에서 백그라운드 태스크를
    시작, 모니터링, 업데이트하기 위한 도구를 추가합니다.
    동기 `SubAgentMiddleware`와 달리, 비동기 서브에이전트는
    즉시 task ID를 반환하여 메인 에이전트가 작업을 계속할 수 있습니다.

    Agent Protocol 호환 서버라면 어디든 작동합니다 — LangGraph Platform(관리형)
    또는 자체 호스팅(예: Agent Protocol 스펙을 구현하는 FastAPI 서버).

    태스크 ID는 에이전트 상태의 `async_tasks`에 저장되어 컨텍스트
    압축/오프로딩 이후에도 유지되며, 프로그래밍 방식으로 접근할 수 있습니다.

    Args:
        async_subagents: 비동기 서브에이전트 설정 스펙 리스트.
            각 항목에 `name`, `description`, `graph_id`가 필수입니다.
            `url`은 선택 — 생략하면 로컬 서버에 ASGI 전송을 사용합니다.
        system_prompt: 비동기 서브에이전트 도구 사용법에 대해
            메인 에이전트의 시스템 프롬프트에 추가되는 지침.

    사용 예시:
        ```python
        from deepagents.middleware.async_subagents import AsyncSubAgentMiddleware

        middleware = AsyncSubAgentMiddleware(
            async_subagents=[
                {
                    "name": "researcher",
                    "description": "Research agent for deep analysis",
                    "url": "https://my-deployment.langsmith.dev",
                    "graph_id": "research_agent",
                }
            ],
        )
        ```
    """

    # 이 미들웨어가 에이전트 상태에 추가하는 필드 스키마
    state_schema = AsyncSubAgentState

    def __init__(
        self,
        *,
        async_subagents: list[AsyncSubAgent],
        system_prompt: str | None = ASYNC_TASK_SYSTEM_PROMPT,
    ) -> None:
        """AsyncSubAgentMiddleware를 초기화합니다.

        서브에이전트 스펙을 검증하고, 5개의 비동기 태스크 도구를 생성하며,
        시스템 프롬프트에 사용 가능한 에이전트 목록을 추가합니다.

        Args:
            async_subagents: 비동기 서브에이전트 설정 리스트 (최소 1개 필수).
            system_prompt: 도구 사용 지침 (None이면 시스템 프롬프트 주입 건너뜀).

        Raises:
            ValueError: 서브에이전트가 비어있거나 이름이 중복된 경우.
        """
        super().__init__()

        # 최소 1개의 서브에이전트 필수
        if not async_subagents:
            msg = "At least one async subagent must be specified"
            raise ValueError(msg)

        # 이름 중복 검사
        names = [a["name"] for a in async_subagents]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            msg = f"Duplicate async subagent names: {dupes}"
            raise ValueError(msg)

        # 5개의 비동기 태스크 도구 생성
        self.tools = _build_async_subagent_tools(async_subagents)

        # 시스템 프롬프트에 사용 가능한 에이전트 목록 추가
        if system_prompt:
            agents_desc = "\n".join(f"- {a['name']}: {a['description']}" for a in async_subagents)
            self.system_prompt: str | None = system_prompt + "\n\nAvailable async subagent types:\n" + agents_desc
        else:
            self.system_prompt = system_prompt

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        """시스템 메시지에 비동기 서브에이전트 사용 지침을 주입합니다 (동기 버전).

        매 LLM 호출 전에 비동기 태스크 도구의 워크플로우와 규칙을
        시스템 프롬프트에 추가합니다.

        Args:
            request: 처리 중인 모델 요청.
            handler: 수정된 요청으로 호출할 핸들러 함수.

        Returns:
            핸들러로부터 받은 모델 응답.
        """
        if self.system_prompt is not None:
            new_system_message = append_to_system_message(request.system_message, self.system_prompt)
            return handler(request.override(system_message=new_system_message))
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT]:
        """시스템 메시지에 비동기 서브에이전트 사용 지침을 주입합니다 (비동기 버전).

        동기 버전과 동일한 로직이지만, 비동기 핸들러를 await합니다.

        Args:
            request: 처리 중인 모델 요청.
            handler: 수정된 요청으로 호출할 비동기 핸들러 함수.

        Returns:
            핸들러로부터 받은 모델 응답.
        """
        if self.system_prompt is not None:
            new_system_message = append_to_system_message(request.system_message, self.system_prompt)
            return await handler(request.override(system_message=new_system_message))
        return await handler(request)
