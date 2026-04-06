from __future__ import annotations

import sys
import types

from deepeval.test_case import ToolCall

from src.loop2_evaluation.agent_runner import (
    extract_final_output,
    extract_tools_called,
    load_available_tools,
    load_agent_from_module,
)


def test_load_agent_from_module_reads_global_agent(monkeypatch):
    module = types.ModuleType("tmp_test_agent_module")
    module.agent = object()
    monkeypatch.setitem(sys.modules, "tmp_test_agent_module", module)

    loaded = load_agent_from_module("tmp_test_agent_module")
    assert loaded is module.agent


def test_extract_final_output_reads_last_ai_message():
    response = {
        "messages": [
            {"role": "user", "content": "질문"},
            {"role": "assistant", "content": "중간 답변"},
            {"role": "assistant", "content": "최종 답변"},
        ]
    }

    assert extract_final_output(response) == "최종 답변"


def test_extract_tools_called_returns_none_when_missing():
    response = {
        "messages": [
            {"role": "user", "content": "질문"},
            {"role": "assistant", "content": "도구 없이 답변"},
        ]
    }

    assert extract_tools_called(response) is None


def test_extract_tools_called_parses_tool_calls_and_outputs():
    response = {
        "messages": [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "name": "search_web",
                        "args": {"query": "SLA"},
                    }
                ],
            },
            {
                "type": "tool",
                "tool_call_id": "call-1",
                "content": "검색 결과: SLA 정보",
            },
            {
                "role": "assistant",
                "content": "SLA는 서비스 수준 합의입니다.",
            },
        ]
    }

    tools_called = extract_tools_called(response)

    assert tools_called is not None
    assert len(tools_called) == 1
    assert isinstance(tools_called[0], ToolCall)
    assert tools_called[0].name == "search_web"
    assert tools_called[0].input_parameters == {"query": "SLA"}
    assert tools_called[0].output == "검색 결과: SLA 정보"


def test_load_available_tools_reads_agent_tools(monkeypatch):
    module = types.ModuleType("tmp_tools_agent_module")
    module.agent = object()
    module.agent_tools = [
        types.SimpleNamespace(name="search_web", description="search the web"),
        types.SimpleNamespace(name="lookup_docs", description="lookup docs"),
    ]
    monkeypatch.setitem(sys.modules, "tmp_tools_agent_module", module)

    tools = load_available_tools("tmp_tools_agent_module")

    assert [tool.name for tool in tools] == ["search_web", "lookup_docs"]
    assert tools[0].description == "search the web"
