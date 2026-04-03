"""Step 5용 실제 agent 실행 어댑터.

Loop 2 오프라인 평가는 이제 Golden의 정답을 그대로 비교하지 않고,
실제 agent를 호출해 얻은 응답을 `actual_output`으로 평가합니다.

이 모듈의 책임:
1. agent 모듈 import
2. 전역 `agent` 변수 검증
3. Golden input으로 agent.invoke() 호출
4. 최종 답변 추출
5. tool call trajectory를 DeepEval ToolCall 형식으로 변환
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from typing import Any

from deepeval.test_case import ToolCall
from langfuse.langchain import CallbackHandler

from src.observability.langfuse import build_langchain_config, enabled
from src.settings import get_settings


@dataclass
class AgentExecutionResult:
    """Step 5 평가기에서 사용할 실제 agent 실행 결과."""

    actual_output: str
    tools_called: list[ToolCall] | None
    raw_response: Any


def load_agent_from_module(agent_module: str):
    """agent 모듈을 import하고 전역 `agent`를 반환합니다."""
    module = importlib.import_module(agent_module)
    agent = getattr(module, "agent", None)
    if agent is None:
        raise ValueError(f"Module '{agent_module}' does not export a global 'agent'")
    return agent


def load_available_tools(agent_module: str) -> list[ToolCall]:
    """agent 모듈의 `agent_tools`에서 사용 가능한 도구 목록을 읽습니다."""
    module = importlib.import_module(agent_module)
    raw_tools = getattr(module, "agent_tools", None)
    if not raw_tools:
        return []

    available_tools: list[ToolCall] = []
    for tool in raw_tools:
        name = getattr(tool, "name", None)
        if not name:
            continue
        available_tools.append(
            ToolCall(
                name=str(name),
                description=getattr(tool, "description", None),
            )
        )

    return available_tools


def _message_attr(message: Any, key: str, default: Any = None) -> Any:
    if isinstance(message, dict):
        return message.get(key, default)
    return getattr(message, key, default)


def _normalize_content(content: Any) -> str:
    """LangChain message content를 사람이 읽을 문자열로 정규화합니다."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    parts.append(item["content"])
        return "\n".join(part for part in parts if part).strip()
    return str(content)


def _message_type(message: Any) -> str:
    if isinstance(message, dict):
        role = message.get("role")
        if role:
            return str(role).lower()
        return str(message.get("type", "")).lower()

    role = getattr(message, "role", None)
    if role:
        return str(role).lower()
    msg_type = getattr(message, "type", None)
    if msg_type:
        return str(msg_type).lower()
    return message.__class__.__name__.lower()


def extract_final_output(response: Any) -> str:
    """agent 응답에서 마지막 AI/assistant 메시지의 content를 추출합니다."""
    if isinstance(response, dict):
        messages = response.get("messages", [])
    else:
        messages = getattr(response, "messages", [])

    for message in reversed(messages or []):
        msg_type = _message_type(message)
        if msg_type in {"ai", "assistant", "aimessage"}:
            content = _normalize_content(_message_attr(message, "content", ""))
            if content:
                return content

    raise ValueError("No final AI message content found in agent response")


def _parse_tool_input_parameters(raw_args: Any) -> dict[str, Any] | None:
    if raw_args is None:
        return None
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args)
            return parsed if isinstance(parsed, dict) else {"value": parsed}
        except Exception:
            return {"raw": raw_args}
    return {"value": raw_args}


def extract_tools_called(response: Any) -> list[ToolCall] | None:
    """agent 응답의 messages에서 tool call 흔적을 DeepEval ToolCall로 변환합니다."""
    if isinstance(response, dict):
        messages = response.get("messages", [])
    else:
        messages = getattr(response, "messages", [])

    if not messages:
        return None

    tool_outputs_by_id: dict[str, Any] = {}
    for message in messages:
        msg_type = _message_type(message)
        if msg_type not in {"tool", "toolmessage"}:
            continue

        tool_call_id = _message_attr(message, "tool_call_id", None) or _message_attr(
            message, "id", None
        )
        if tool_call_id:
            tool_outputs_by_id[str(tool_call_id)] = _normalize_content(
                _message_attr(message, "content", "")
            )

    tools_called: list[ToolCall] = []
    for message in messages:
        tool_calls = _message_attr(message, "tool_calls", None)
        if not tool_calls:
            continue

        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue

            tool_id = tool_call.get("id")
            tools_called.append(
                ToolCall(
                    name=str(tool_call.get("name", "unknown_tool")),
                    input_parameters=_parse_tool_input_parameters(tool_call.get("args")),
                    output=tool_outputs_by_id.get(str(tool_id)) if tool_id is not None else None,
                )
            )

    return tools_called or None


def execute_agent_on_input(
    *,
    agent_module: str,
    user_input: str,
    item_id: str | None = None,
) -> AgentExecutionResult:
    """Golden input 하나를 실제 agent에 넣고 결과를 반환합니다."""
    agent = load_agent_from_module(agent_module)
    settings = get_settings()

    invoke_input = {"messages": [{"role": "user", "content": user_input}]}
    invoke_config: dict[str, Any] | None = None

    if enabled(settings):
        # Langfuse는 Step 5의 핵심 결과물은 아니지만,
        # 실제 agent 평가 시 어떤 입력이 어떻게 실행됐는지 추적하는 데 유용합니다.
        invoke_config = build_langchain_config(
            user_id="step5-evaluator",
            session_id=f"step5-{item_id or 'unknown'}",
            callbacks=[CallbackHandler()],
            tags=["step5", "agent-eval", agent_module],
            settings=settings,
        )

    response = (
        agent.invoke(invoke_input, config=invoke_config)
        if invoke_config is not None
        else agent.invoke(invoke_input)
    )

    return AgentExecutionResult(
        actual_output=extract_final_output(response),
        tools_called=extract_tools_called(response),
        raw_response=response,
    )
