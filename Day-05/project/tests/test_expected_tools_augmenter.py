from __future__ import annotations

from types import SimpleNamespace


def test_augment_expected_tools_adds_expected_tools(monkeypatch):
    from src.loop1_dataset.expected_tools_augmenter import augment_expected_tools

    class FakeCompletions:
        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=(
                                '{"expected_tools":[{"name":"search_web",'
                                '"input_parameters":{"query":"SLA metrics"},'
                                '"reasoning":"Need external information"}]}'
                            )
                        )
                    )
                ]
            )

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))

    monkeypatch.setattr(
        "src.loop1_dataset.expected_tools_augmenter.get_settings",
        lambda: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "src.loop1_dataset.expected_tools_augmenter.get_llm_client",
        lambda settings=None: fake_client,
    )
    monkeypatch.setattr(
        "src.loop1_dataset.expected_tools_augmenter.get_chat_model_name",
        lambda settings=None: "gpt-5.4-mini",
    )
    monkeypatch.setattr(
        "src.loop1_dataset.expected_tools_augmenter.load_agent_tools_for_dataset",
        lambda agent_module: [{"name": "search_web", "description": "search the web"}],
    )

    items = augment_expected_tools(
        [
            {
                "id": "g1",
                "input": "SLA는 무엇인가요?",
                "expected_output": "SLA는 서비스 수준 합의입니다.",
                "context": ["SLA context"],
            }
        ],
        agent_module="src.my_agent",
    )

    assert items[0]["expected_tools"][0]["name"] == "search_web"
    assert items[0]["expected_tools"][0]["input_parameters"] == {"query": "SLA metrics"}


def test_augment_expected_tools_returns_empty_when_no_agent_tools(monkeypatch):
    from src.loop1_dataset.expected_tools_augmenter import augment_expected_tools

    monkeypatch.setattr(
        "src.loop1_dataset.expected_tools_augmenter.load_agent_tools_for_dataset",
        lambda agent_module: [],
    )

    items = augment_expected_tools(
        [
            {
                "id": "g1",
                "input": "SLA는 무엇인가요?",
                "expected_output": "SLA는 서비스 수준 합의입니다.",
                "context": ["SLA context"],
            }
        ],
        agent_module="src.my_agent",
    )

    assert items[0]["expected_tools"] == []


def test_augment_expected_tools_uses_search_fallback_when_llm_returns_empty(monkeypatch):
    from src.loop1_dataset.expected_tools_augmenter import augment_expected_tools

    class FakeCompletions:
        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"expected_tools":[]}'))]
            )

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))

    monkeypatch.setattr(
        "src.loop1_dataset.expected_tools_augmenter.get_settings",
        lambda: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "src.loop1_dataset.expected_tools_augmenter.get_llm_client",
        lambda settings=None: fake_client,
    )
    monkeypatch.setattr(
        "src.loop1_dataset.expected_tools_augmenter.get_chat_model_name",
        lambda settings=None: "gpt-5.4-mini",
    )
    monkeypatch.setattr(
        "src.loop1_dataset.expected_tools_augmenter.load_agent_tools_for_dataset",
        lambda agent_module: [{"name": "search_web", "description": "search the web"}],
    )

    items = augment_expected_tools(
        [
            {
                "id": "g1",
                "input": "How should I compare SLA metrics?",
                "expected_output": "Compare availability, latency, and error rate.",
                "context": ["SLA context"],
            }
        ],
        agent_module="src.my_agent",
    )

    assert items[0]["expected_tools"] == [
        {
            "name": "search_web",
            "input_parameters": {"query": "How should I compare SLA metrics?"},
            "reasoning": "Fallback expected tool for an information-seeking question",
        }
    ]
