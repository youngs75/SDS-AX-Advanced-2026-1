import asyncio
import inspect
import json
from collections.abc import AsyncIterable, Awaitable, Callable
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, convert_to_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    DataPart,
    InternalError,
    Part,
    TaskNotFoundError,
    TaskState,
    TextPart,
)
from a2a.utils import get_data_parts, new_agent_text_message, new_task
from a2a.utils.errors import ServerError


class _ExecutorSupport:
    async def _get_or_create_task(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> tuple[Any, bool]:
        task = context.current_task
        if task is not None:
            return task, False

        if context.message is None:
            raise ServerError(error=InternalError(message="요청 메시지가 없습니다."))

        task = new_task(context.message)
        context.current_task = task
        await event_queue.enqueue_event(task)
        return task, True

    def _extract_last_data_payload(
        self, context: RequestContext
    ) -> dict[str, Any] | None:
        if not context.message or not getattr(context.message, "parts", None):
            return None

        try:
            data_parts = get_data_parts(context.message.parts)
        except Exception:
            logger.exception("DataPart 파싱에 실패했습니다.")
            return None

        if not data_parts:
            return None

        payload = data_parts[-1]
        return payload if isinstance(payload, dict) else None

    async def _publish_text_result(self, updater: TaskUpdater, text: str) -> None:
        await updater.add_artifact([Part(root=TextPart(text=text))])

    async def _fail_task(self, updater: TaskUpdater, task: Any, exc: Exception) -> None:
        error_text = f"A2A 실행 중 오류: {exc}"
        message = new_agent_text_message(error_text, task.context_id, task.id)
        await updater.failed(message=message)

    def _is_input_required_task(self, task: Any) -> bool:
        state = getattr(getattr(task, "status", None), "state", None)
        value = getattr(state, "value", state)
        return value == TaskState.input_required.value


class BaseAgentExecutor(_ExecutorSupport, AgentExecutor):
    """일반 callable 기반 에이전트를 A2A executor로 감싼다.

    입력 계약:
    - 기본 입력은 사용자 text 또는 마지막 DataPart payload 중 하나다.
    - callable은 `(agent_input)` 또는 `(agent_input, context)` 형태를 지원한다.

    출력 계약:
    - `str`: 최종 TextPart artifact
    - `dict` / `list`: 최종 DataPart artifact
    - `AsyncIterable[str]`: 증분 working 메시지 + 최종 TextPart artifact
    """

    def __init__(
        self,
        agent: Callable[..., Any],
        result_extractor: Callable[[Any], str] | None = None,
    ) -> None:
        self.agent = agent
        self.result_extractor = result_extractor
        self._cancelled_task_ids: set[str] = set()

    def _call_agent(self, agent_input: Any, context: RequestContext) -> Any:
        signature = inspect.signature(self.agent)
        params = list(signature.parameters.values())
        accepts_varargs = any(
            param.kind == inspect.Parameter.VAR_POSITIONAL for param in params
        )
        positional_params = [
            param
            for param in params
            if param.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]

        if accepts_varargs or len(positional_params) >= 2:
            return self.agent(agent_input, context)
        return self.agent(agent_input)

    async def _consume_async_iterable(
        self,
        result: AsyncIterable[Any],
        updater: TaskUpdater,
        task: Any,
    ) -> str:
        chunks: list[str] = []
        async for chunk in result:
            if str(task.id) in self._cancelled_task_ids:
                break
            if not isinstance(chunk, str):
                raise TypeError("AsyncIterable 결과는 문자열 청크만 지원합니다.")
            chunks.append(chunk)
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(chunk, task.context_id, task.id),
            )
        return "".join(chunks)

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        task: Any | None = None
        task_id: str | None = None

        try:
            task, _ = await self._get_or_create_task(context, event_queue)
            task_id = str(task.id)
            payload = self._extract_last_data_payload(context)
            query = context.get_user_input()
            agent_input: Any = payload if payload is not None else query
            updater = TaskUpdater(event_queue, task.id, task.context_id)

            if task_id in self._cancelled_task_ids:
                return

            await updater.start_work()

            if task_id in self._cancelled_task_ids:
                return

            result = self._call_agent(agent_input, context)
            if isinstance(result, Awaitable):
                result = await result

            if hasattr(result, "__aiter__"):
                result = await self._consume_async_iterable(result, updater, task)

            if task_id in self._cancelled_task_ids:
                return

            if isinstance(result, str):
                await self._publish_text_result(updater, result)
            elif isinstance(result, (dict, list)):
                await updater.add_artifact([Part(root=DataPart(data=result))])
                if self.result_extractor is not None:
                    extracted = self.result_extractor(result)
                    if extracted:
                        await self._publish_text_result(updater, extracted)
            else:
                raise TypeError(
                    "BaseAgentExecutor는 str, dict, list, AsyncIterable[str] 결과만 지원합니다."
                )

            await updater.complete()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if task_id is not None and task_id in self._cancelled_task_ids:
                return
            if task is not None:
                updater = TaskUpdater(event_queue, task.id, task.context_id)
                await self._fail_task(updater, task, exc)
            raise ServerError(error=InternalError()) from exc
        finally:
            if task_id is not None:
                self._cancelled_task_ids.discard(task_id)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        task = context.current_task
        if task is None:
            raise ServerError(error=TaskNotFoundError())

        self._cancelled_task_ids.add(str(task.id))
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        message = new_agent_text_message(
            "사용자의 요청으로 작업이 취소되었습니다.",
            task.context_id,
            task.id,
        )
        await updater.cancel(message)


class LGAgentExecutor(_ExecutorSupport, AgentExecutor):
    """LangGraph CompiledStateGraph를 A2A executor로 감싼다."""

    def __init__(
        self,
        graph: CompiledStateGraph,
        result_extractor: Callable[[Any], str] | None = None,
    ) -> None:
        self.graph = graph
        self._extract_result_text = result_extractor or self._default_extract_text
        self._cancelled_task_ids: set[str] = set()

    def _build_graph_input(
        self,
        payload: dict[str, Any] | None,
        query: str,
    ) -> dict[str, Any]:
        if isinstance(payload, dict) and payload:
            graph_input = dict(payload)
            if "messages" in graph_input and graph_input["messages"] is not None:
                graph_input["messages"] = convert_to_messages(graph_input["messages"])
            return graph_input
        return {"messages": [HumanMessage(content=query)]}

    def _extract_resume_value(
        self,
        payload: dict[str, Any] | None,
        query: str,
    ) -> Any:
        if isinstance(payload, dict):
            for key in ("resume", "answer", "user_input", "value"):
                if payload.get(key) is not None:
                    return payload[key]
        if query.strip():
            return query.strip()
        return None

    def _default_extract_text(self, result: Any) -> str:
        if isinstance(result, dict):
            messages = result.get("messages")
            if isinstance(messages, list):
                ai_messages = [
                    message for message in messages if isinstance(message, AIMessage)
                ]
                if ai_messages:
                    return self._stringify_message_content(ai_messages[-1].content)
        if isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=False, default=str)

    def _stringify_message_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts: list[str] = []
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    texts.append(part["text"])
                else:
                    text_attr = getattr(part, "text", None)
                    if isinstance(text_attr, str):
                        texts.append(text_attr)
            return "".join(texts)
        return str(content)

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        task: Any | None = None
        task_id: str | None = None

        try:
            task, _ = await self._get_or_create_task(context, event_queue)
            task_id = str(task.id)
            payload = self._extract_last_data_payload(context)
            query = context.get_user_input()
            updater = TaskUpdater(event_queue, task.id, task.context_id)

            if task_id in self._cancelled_task_ids:
                return

            if self._is_input_required_task(task):
                invoke_input: Any = Command(
                    resume=self._extract_resume_value(payload, query)
                )
            else:
                invoke_input = self._build_graph_input(payload, query)

            await updater.start_work()

            if task_id in self._cancelled_task_ids:
                return

            config = {"configurable": {"thread_id": task_id}}
            result = await self.graph.ainvoke(invoke_input, config=config)

            if task_id in self._cancelled_task_ids:
                return

            final_text = self._extract_result_text(result)
            if not final_text:
                final_text = "결과 텍스트를 생성하지 못했습니다."

            await self._publish_text_result(updater, final_text)
            await updater.complete()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if task_id is not None and task_id in self._cancelled_task_ids:
                return
            if task is not None:
                updater = TaskUpdater(event_queue, task.id, task.context_id)
                await self._fail_task(updater, task, exc)
            raise ServerError(error=InternalError()) from exc
        finally:
            if task_id is not None:
                self._cancelled_task_ids.discard(task_id)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        task = context.current_task
        if task is None:
            raise ServerError(error=TaskNotFoundError())

        self._cancelled_task_ids.add(str(task.id))
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        message = new_agent_text_message(
            "사용자의 요청으로 작업이 취소되었습니다.",
            task.context_id,
            task.id,
        )
        await updater.cancel(message)
