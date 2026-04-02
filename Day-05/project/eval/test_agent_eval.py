"""CI Gate: deepeval test run eval/test_agent_eval.py

DeepEval의 pytest 통합을 통해 CI 파이프라인에서 자동 평가를 수행합니다.
`deepeval test run eval/test_agent_eval.py` 명령으로 실행합니다.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase

from src.llm.deepeval_model import get_deepeval_model
from src.loop2_evaluation.rag_metrics import create_rag_metrics
from src.loop2_evaluation.custom_metrics import create_response_completeness_metric
from src.settings import get_settings


def _load_golden_dataset() -> list[dict]:
    """Golden Dataset 로드."""
    settings = get_settings()
    golden_path = settings.data_dir / "golden" / "golden_dataset.json"
    if not golden_path.exists():
        pytest.skip(f"Golden dataset not found: {golden_path}")
    with open(golden_path, encoding="utf-8") as f:
        return json.load(f)


def _make_test_cases() -> list[LLMTestCase]:
    """Golden Dataset → LLMTestCase 변환."""
    golden_data = _load_golden_dataset()
    test_cases = []
    for item in golden_data:
        tc = LLMTestCase(
            input=item.get("input", ""),
            actual_output=item.get("expected_output", ""),
            expected_output=item.get("expected_output", ""),
            context=item.get("context") if item.get("context") else None,
            retrieval_context=(
                item.get("retrieval_context")
                if item.get("retrieval_context")
                else item.get("context")
            ),
        )
        test_cases.append(tc)
    return test_cases


@pytest.mark.parametrize("test_case", _make_test_cases() if Path("data/golden/golden_dataset.json").exists() else [], ids=lambda tc: tc.input[:50])
def test_rag_quality(test_case: LLMTestCase):
    """RAG 품질 게이트: AnswerRelevancy + Faithfulness."""
    metrics = create_rag_metrics(
        relevancy_threshold=0.7,
        faithfulness_threshold=0.7,
        precision_threshold=0.5,
        recall_threshold=0.5,
    )
    for metric in metrics:
        assert_test(test_case, [metric])


@pytest.mark.parametrize("test_case", _make_test_cases() if Path("data/golden/golden_dataset.json").exists() else [], ids=lambda tc: tc.input[:50])
def test_response_completeness(test_case: LLMTestCase):
    """응답 완전성 게이트."""
    metric = create_response_completeness_metric(threshold=0.7)
    assert_test(test_case, [metric])
