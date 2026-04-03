"""Golden Dataset의 expected_tools 보강 모듈.

ToolCorrectnessMetric은 `tools_called`와 `expected_tools`를 모두 필요로 합니다.
이 모듈은 Loop 1에서 Golden 항목을 확정할 때,
평가 대상 agent의 도구 목록을 기준으로 expected_tools를 미리 채워 넣습니다.
"""

from __future__ import annotations

import json
from importlib import import_module
from typing import Any

from pydantic import BaseModel, Field

from src.llm.openrouter import get_chat_model_name, get_llm_client
from src.loop1_dataset.prompts import EXPECTED_TOOLS_PROMPT
from src.settings import get_settings


class ExpectedTool(BaseModel):
    name: str
    input_parameters: dict[str, Any] | None = None
    reasoning: str | None = None


class ExpectedToolsResponse(BaseModel):
    expected_tools: list[ExpectedTool] = Field(default_factory=list)


def load_agent_tools_for_dataset(agent_module: str) -> list[dict[str, str]]:
    """agent 모듈의 `agent_tools`를 Golden 생성용 메타데이터로 변환합니다."""
    module = import_module(agent_module)
    raw_tools = getattr(module, "agent_tools", None)
    if not raw_tools:
        return []

    tools: list[dict[str, str]] = []
    for tool in raw_tools:
        name = getattr(tool, "name", None)
        if not name:
            continue
        tools.append(
            {
                "name": str(name),
                "description": str(getattr(tool, "description", "") or ""),
            }
        )
    return tools


def _fallback_expected_tools(
    *,
    user_input: str,
    available_tools: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """LLM이 빈 expected_tools를 반환했을 때 적용하는 보수적 fallback.

    검색/조회 계열 도구가 있고, 질문이 정보 탐색형이면
    최소한 그 도구 하나는 기대 도구로 남겨 ToolCorrectnessMetric이 작동하게 합니다.
    """
    lowered_input = user_input.lower()
    search_like_keywords = ("search", "web", "lookup", "retrieve", "find")
    info_seeking_markers = (
        "?",
        "what",
        "how",
        "compare",
        "which",
        "define",
        "why",
        "설명",
        "비교",
        "무엇",
        "어떻게",
    )

    if not any(marker in lowered_input or marker in user_input for marker in info_seeking_markers):
        return []

    for tool in available_tools:
        name = tool.get("name", "")
        description = tool.get("description", "")
        haystack = f"{name} {description}".lower()
        if any(keyword in haystack for keyword in search_like_keywords):
            return [
                {
                    "name": name,
                    "input_parameters": {"query": user_input},
                    "reasoning": "Fallback expected tool for an information-seeking question",
                }
            ]

    return []


def _context_to_text(context: Any) -> str:
    if isinstance(context, list):
        return "\n".join(str(item) for item in context)
    return str(context or "")


def augment_expected_tools(
    items: list[dict],
    *,
    agent_module: str,
) -> list[dict]:
    """Golden 항목마다 expected_tools를 추론해 추가합니다."""
    available_tools = load_agent_tools_for_dataset(agent_module)
    if not available_tools:
        return [{**item, "expected_tools": []} for item in items]

    settings = get_settings()
    client = get_llm_client(settings=settings)
    model_name = get_chat_model_name(settings)

    augmented_items: list[dict] = []
    tools_json = json.dumps(available_tools, ensure_ascii=False, indent=2)

    for item in items:
        prompt = EXPECTED_TOOLS_PROMPT.format(
            input=item.get("input", ""),
            expected_output=item.get("expected_output", ""),
            context=_context_to_text(item.get("context")),
            available_tools=tools_json,
        )

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            content = response.choices[0].message.content or ""
            parsed = ExpectedToolsResponse.model_validate_json(content)
            expected_tools = [
                expected_tool.model_dump(exclude_none=True)
                for expected_tool in parsed.expected_tools
            ]
            if not expected_tools:
                expected_tools = _fallback_expected_tools(
                    user_input=str(item.get("input", "")),
                    available_tools=available_tools,
                )
            augmented_items.append(
                {
                    **item,
                    "expected_tools": expected_tools,
                }
            )
        except Exception as exc:
            print(
                f"[WARN] Expected tools augmentation failed for '{item.get('id', 'unknown')}': {exc}"
            )
            augmented_items.append({**item, "expected_tools": []})

    return augmented_items
