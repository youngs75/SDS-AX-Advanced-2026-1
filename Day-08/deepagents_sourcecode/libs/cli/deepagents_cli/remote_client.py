"""원격 에이전트 클라이언트 — LangGraph의 `RemoteGraph`을 둘러싼 씬 래퍼입니다.

스트리밍, 상태 관리 및 SSE 처리를 `langgraph.pregel.remote.RemoteGraph`에 위임합니다. 추가된 유일한 논리는 서버의 원시
메시지 구문을 CLI의 텍스트 어댑터가 예상하는 LangChain 메시지 개체로 변환하는 것입니다.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

from deepagents_cli._debug import configure_debug_logging

logger = logging.getLogger(__name__)
configure_debug_logging(logger)


def _require_thread_id(config: dict[str, Any] | None) -> str:
    """`thread_id`이 구성에 있는지 추출하고 검증합니다.

Args:
        config: `configurable.thread_id`을(를) 사용하여 사전을 구성합니다.

Returns:
        스레드 ID 문자열입니다.

Raises:
        ValueError: `thread_id`이 누락된 경우.

    """
    thread_id = (config or {}).get("configurable", {}).get("thread_id")
    if not thread_id:
        msg = "thread_id is required in config.configurable"
        raise ValueError(msg)
    return thread_id


class RemoteAgent:
    """HTTP+SSE를 통해 LangGraph 서버와 통신하는 클라이언트입니다.

    SSE 구문 분석, 스트림 모드 협상(`messages-tuple`), 네임스페이스 추출 및 인터럽트 감지를 처리하는
    `langgraph.pregel.remote.RemoteGraph`을 래핑합니다. 이 클래스는 텍스트 어댑터 및 스레드 ID 정규화를 위한 메시지-객체
    변환만 추가합니다.

    """

    def __init__(
        self,
        url: str,
        *,
        graph_name: str = "agent",
        api_key: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        """원격 에이전트 클라이언트를 초기화합니다.

Args:
            url: LangGraph 서버의 기본 URL입니다.
            graph_name: 서버의 그래프 이름입니다.
            api_key: 인증된 배포를 위한 API 키입니다.

                `None`, `RemoteGraph`이(가) 환경에서 `LANGGRAPH_API_KEY`, `LANGSMITH_API_KEY`
                또는 `LANGCHAIN_API_KEY`을 자동으로 읽는 경우.
            headers: 모든 요청에 ​​포함할 추가 HTTP 헤더(예: 전달자 토큰, 프록시 헤더)

        """
        self._url = url
        self._graph_name = graph_name
        self._api_key = api_key
        self._headers = headers
        self._graph: Any = None

    def _get_graph(self) -> Any:  # noqa: ANN401
        """`RemoteGraph` 인스턴스를 느리게 생성합니다.

Returns:
            `RemoteGraph`이 서버에 연결되었습니다.

        """
        if self._graph is None:
            from langgraph.pregel.remote import RemoteGraph

            self._graph = RemoteGraph(
                self._graph_name,
                url=self._url,
                api_key=self._api_key,
                headers=self._headers,
            )
        return self._graph

    async def astream(
        self,
        input: dict | Any,  # noqa: A002, ANN401
        *,
        stream_mode: list[str] | None = None,
        subgraphs: bool = False,
        config: dict[str, Any] | None = None,
        context: Any | None = None,  # noqa: ANN401
        durability: str | None = None,  # noqa: ARG002
    ) -> AsyncIterator[tuple[tuple[str, ...], str, Any]]:
        """스트림 에이전트 실행으로 Pregel 형식과 일치하는 튜플을 생성합니다.

        `RemoteGraph.astream`(`messages-tuple` 협상, SSE 라우팅 및 네임스페이스 구문 분석을 처리)에 위임하고 원시
        메시지 사전을 어댑터의 LangChain 메시지 개체로 변환합니다.

Args:
            input: 보낼 입력(메시지 dict 또는 Command)입니다.
            stream_mode: 요청할 스트림 모드.
            subgraphs: 하위 그래프 이벤트를 스트리밍할지 여부입니다.
            config: `configurable.thread_id` 등을 사용한 LangGraph 구성
            context: SDK의 `context=` 매개변수를 통해 서버로 전달되는 런타임 컨텍스트(예: `CLIContext`).
            durability: 무시됩니다(서버가 내구성을 관리함).

Yields:
            `(namespace, stream_mode, data)`의 3튜플.

Raises:
            ValueError: `thread_id`이(가) `config`에 없는 경우.

        """  # noqa: DOC502 — raised by _require_thread_id
        from langchain_core.messages import BaseMessage

        _require_thread_id(config)

        graph = self._get_graph()
        config = _prepare_config(config)
        dropped_count = 0

        async for ns, mode, data in graph.astream(
            input,
            stream_mode=stream_mode or ["messages", "updates"],
            subgraphs=subgraphs,
            config=config,
            context=context,
        ):
            logger.debug("RemoteGraph event mode=%s ns=%s", mode, ns)

            if mode == "messages":
                msg_dict, meta = data
                if isinstance(msg_dict, dict):
                    msg_obj = _convert_message_data(msg_dict)
                    if msg_obj is not None:
                        yield (ns, "messages", (msg_obj, meta or {}))
                    else:
                        dropped_count += 1
                elif isinstance(msg_dict, BaseMessage):
                    # Already a LangChain message object (pre-deserialized)
                    yield (ns, "messages", (msg_dict, meta or {}))
                else:
                    logger.warning(
                        "Unexpected message data type in stream: %s",
                        type(msg_dict).__name__,
                    )
                continue

            if mode == "updates" and isinstance(data, dict):
                update_data = data
                if "__interrupt__" in data:
                    update_data = {
                        **data,
                        "__interrupt__": _convert_interrupts(data["__interrupt__"]),
                    }
                yield (ns, "updates", update_data)
                continue

            yield (ns, mode, data)

        if dropped_count:
            logger.warning(
                "Dropped %d message(s) during stream due to conversion failures",
                dropped_count,
            )

    async def aget_state(
        self,
        config: dict[str, Any],
    ) -> Any:  # noqa: ANN401
        """스레드의 현재 상태를 가져옵니다.

        서버에 스레드가 없으면 `None`을 반환합니다(404). 다른 모든 오류(네트워크, 인증, 500)는 WARNING에 기록되고 호출자가 처리할
        수 있도록 다시 발생합니다.

Args:
            config: `configurable.thread_id`로 구성하세요.

Returns:
            `values` 및 `next` 속성 ​​또는 `None`이 있는 스레드 상태 객체
                스레드를 찾을 수 없는 경우.

Raises:
            ValueError: `thread_id`이(가) `config`에 없는 경우.

        """  # noqa: DOC502 — raised by _require_thread_id
        from langgraph_sdk.errors import NotFoundError

        thread_id = _require_thread_id(config)

        graph = self._get_graph()
        try:
            return await graph.aget_state(_prepare_config(config))
        except NotFoundError:
            logger.debug("Thread %s not found on server", thread_id)
            return None
        except Exception:
            logger.warning(
                "Failed to get state for thread %s", thread_id, exc_info=True
            )
            raise

    async def aupdate_state(
        self,
        config: dict[str, Any],
        values: dict[str, Any],
    ) -> None:
        """스레드 상태를 업데이트합니다.

        기본 그래프의 예외(서버/네트워크 오류)는 WARNING 수준에서 기록된 다음 다시 발생하여 호출자가 이를 처리할 수 있습니다.

Args:
            config: `configurable.thread_id`로 구성하세요.
            values: 업데이트할 상태 값입니다.

Raises:
            ValueError: `thread_id`이(가) `config`에 없는 경우.

        """  # noqa: DOC502 — raised by _require_thread_id
        thread_id = _require_thread_id(config)

        graph = self._get_graph()
        try:
            await graph.aupdate_state(_prepare_config(config), values)
        except Exception:
            logger.warning(
                "Failed to update state for thread %s", thread_id, exc_info=True
            )
            raise

    async def aensure_thread(self, config: dict[str, Any]) -> None:
        """상태를 변경하기 전에 원격 스레드 레코드가 존재하는지 확인하세요.

        LangGraph 개발 서버에서는 체크포인트 지속성과 HTTP 스레드 등록이 별개입니다. 서버를 다시 시작한 후에도 스레드는 여전히 디스크에
        체크포인트 상태를 가질 수 있지만 서버가 해당 스레드를 라이브 저장소에 아직 구체화하지 않았기 때문에 `POST
        /threads/{id}/state`은 404를 반환합니다.

        이 방법은 `if_exists='do_nothing'`을 사용하여 멱등성 HTTP 측 등록을 수행하므로 지속성에서 상태를 복구한 호출자가
        `aupdate_state`을(를) 안전하게 후속 조치할 수 있습니다.

Args:
            config: `configurable.thread_id` 및 선택적 메타데이터로 구성합니다.

Raises:
            ValueError: `thread_id`이(가) `config`에 없는 경우.

        """  # noqa: DOC502 — raised by _require_thread_id
        _require_thread_id(config)

        graph = self._get_graph()
        prepared = _prepare_config(config)
        thread_id = prepared["configurable"]["thread_id"]
        metadata = prepared.get("metadata")
        thread_metadata = metadata if isinstance(metadata, dict) else None

        try:
            client = graph._validate_client()
            await client.threads.create(
                thread_id=thread_id,
                if_exists="do_nothing",
                metadata=thread_metadata,
                graph_id=self._graph_name,
            )
        except Exception:
            logger.warning(
                "Failed to ensure thread %s exists on remote server",
                thread_id,
                exc_info=True,
            )
            raise

    def with_config(self, config: dict[str, Any]) -> RemoteAgent:  # noqa: ARG002
        """자체를 반환합니다(구성은 저장되지 않고 호출별로 전달됩니다).

Args:
            config: 무시되었습니다.

Returns:
            본인.

        """
        return self


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _prepare_config(config: dict[str, Any] | None) -> dict[str, Any]:
    """호출자의 dicts가 변경되지 않도록 얕은 복사 구성입니다.

Args:
        config: 원시 구성 사전

Returns:
        구성의 얕은 복사본입니다.

    """
    config = dict(config or {})
    configurable = dict(config.get("configurable", {}))
    config["configurable"] = configurable
    return config


def _convert_interrupts(raw: Any) -> list[Any]:  # noqa: ANN401
    """서버의 인터럽트 dict를 Interrupt 객체로 변환합니다.

Args:
        raw: 서버의 인터럽트 dict 또는 인터럽트 개체 목록입니다.

Returns:
        인터럽트 개체 목록입니다.

    """
    from langgraph.types import Interrupt

    if not isinstance(raw, list):
        logger.warning(
            "Expected list for __interrupt__ data, got %s",
            type(raw).__name__,
        )
        return [raw] if raw is not None else []
    results = []
    for item in raw:
        if isinstance(item, Interrupt):
            results.append(item)
        elif isinstance(item, dict) and "value" in item:
            results.append(Interrupt(value=item["value"], id=item.get("id", "")))
        else:
            results.append(item)
    return results


# ---------------------------------------------------------------------------
# Message conversion — per-type converters with a dispatch table
# ---------------------------------------------------------------------------
#
# Each converter handles one LangChain message type.  The dispatch table
# maps type strings (both short and class-name forms) to the appropriate
# converter.  This keeps each converter focused and makes adding new
# message types a one-line addition to the table.
# ---------------------------------------------------------------------------


def _convert_ai_message(data: dict[str, Any]) -> Any:  # noqa: ANN401
    """서버 AI 메시지 사전을 `AIMessageChunk`로 변환합니다.

    서버가 내보낼 수 있는 세 가지 도구 호출 표현을 처리합니다.

    - `tool_call_chunks`: 부분 인수 스트리밍(문자열 `args`). - `tool_calls`(문자열 `args` 포함): 레거시
    스트리밍 형식,
        `tool_call_chunks`로 정규화되었습니다.
    - `tool_calls`(딕셔너리 `args` 포함): 완전히 구문 분석된 호출.

Args:
        data: 서버의 원시 메시지 dict입니다.

Returns:
        건설 실패 시 `AIMessageChunk` 또는 `None`.

    """
    from langchain_core.messages import AIMessageChunk

    content = data.get("content", "")
    tool_call_chunks = data.get("tool_call_chunks", [])
    tool_calls = data.get("tool_calls", [])
    usage_metadata = data.get("usage_metadata")
    response_metadata = data.get("response_metadata", {})

    kwargs: dict[str, Any] = {
        "content": content,
        "id": data.get("id"),
        "response_metadata": response_metadata,
    }

    if tool_call_chunks:
        kwargs["tool_call_chunks"] = [
            {
                "name": tc.get("name"),
                "args": tc.get("args", ""),
                "id": tc.get("id"),
                "index": tc.get("index", i),
            }
            for i, tc in enumerate(tool_call_chunks)
        ]
    elif tool_calls:
        has_str_args = any(isinstance(tc.get("args"), str) for tc in tool_calls)
        if has_str_args:
            kwargs["tool_call_chunks"] = [
                {
                    "name": tc.get("name"),
                    "args": tc.get("args", ""),
                    "id": tc.get("id"),
                    "index": i,
                }
                for i, tc in enumerate(tool_calls)
            ]
        else:
            kwargs["tool_calls"] = tool_calls

    try:
        chunk = AIMessageChunk(**kwargs)
    except (TypeError, ValueError, KeyError):
        logger.warning(
            "Failed to construct AIMessageChunk from server data (id=%s)",
            data.get("id"),
            exc_info=True,
        )
        return None

    if usage_metadata:
        chunk.usage_metadata = usage_metadata
    return chunk


def _convert_human_message(data: dict[str, Any]) -> Any:  # noqa: ANN401
    """서버 사람 메시지 사전을 `HumanMessage`로 변환합니다.

Args:
        data: 서버의 원시 메시지 dict입니다.

Returns:
        건설 실패 시 `HumanMessage` 또는 `None`.

    """
    from langchain_core.messages import HumanMessage

    try:
        return HumanMessage(
            content=data.get("content", ""),
            id=data.get("id"),
        )
    except (TypeError, ValueError, KeyError):
        logger.warning(
            "Failed to construct HumanMessage from server data (id=%s)",
            data.get("id"),
            exc_info=True,
        )
        return None


def _convert_tool_message(data: dict[str, Any]) -> Any:  # noqa: ANN401
    """서버 도구 메시지 사전을 `ToolMessage`로 변환합니다.

Args:
        data: 서버의 원시 메시지 dict입니다.

Returns:
        건설 실패 시 `ToolMessage` 또는 `None`.

    """
    from langchain_core.messages import ToolMessage

    try:
        return ToolMessage(
            content=data.get("content", ""),
            tool_call_id=data.get("tool_call_id", ""),
            name=data.get("name", ""),
            id=data.get("id"),
            status=data.get("status", "success"),
        )
    except (TypeError, ValueError, KeyError):
        logger.warning(
            "Failed to construct ToolMessage from server data (id=%s)",
            data.get("id"),
            exc_info=True,
        )
        return None


_MESSAGE_CONVERTERS: dict[str, Callable[[dict[str, Any]], Any]] = {
    "ai": _convert_ai_message,
    "AIMessage": _convert_ai_message,
    "AIMessageChunk": _convert_ai_message,
    "human": _convert_human_message,
    "HumanMessage": _convert_human_message,
    "tool": _convert_tool_message,
    "ToolMessage": _convert_tool_message,
}
"""서버 메시지 `type` 문자열을 해당 변환기 기능에 매핑합니다.

짧은 형식(`'ai'`, `'human'`, `'tool'`)과 클래스 이름 형식(`'AIMessage'`, `'HumanMessage'`,
`'ToolMessage'`)이 모두 지원되므로 서버가 유형 필드를 직렬화하는 방법에 관계없이 변환기가 작동합니다.
"""


def _convert_message_data(data: dict[str, Any]) -> Any:  # noqa: ANN401
    """서버 메시지 사전을 LangChain 메시지 개체로 변환합니다.

    `_MESSAGE_CONVERTERS`을 통해 유형별 변환기로 전달됩니다. 변환기 기능과 테이블 항목을 추가하여 새로운 메시지 유형을 지원할 수
    있습니다. 이 디스패처를 변경할 필요가 없습니다.

Args:
        data: 서버에서 보낸 메시지입니다.

Returns:
        LangChain 메시지 개체 또는 변환이 실패한 경우 `None`.

    """
    msg_type = data.get("type", "")
    converter = _MESSAGE_CONVERTERS.get(msg_type)
    if converter is not None:
        return converter(data)
    logger.warning("Unknown message type in stream: %s", msg_type)
    return None
