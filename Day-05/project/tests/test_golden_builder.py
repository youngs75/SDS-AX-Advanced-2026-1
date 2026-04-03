from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


def test_build_golden_dataset_when_skip_review_then_auto_validates_and_saves(
    tmp_data_dir, monkeypatch
):
    from src.loop1_dataset.golden_builder import build_golden_dataset

    corpus_dir = tmp_data_dir / "corpus"
    corpus_dir.mkdir(exist_ok=True)

    monkeypatch.setattr(
        "src.loop1_dataset.golden_builder.get_settings",
        lambda: SimpleNamespace(data_dir=tmp_data_dir, local_corpus_dir=corpus_dir),
    )
    monkeypatch.setattr(
        "src.loop1_dataset.golden_builder.augment_expected_tools",
        lambda items, agent_module: items,
    )

    candidate_items = [
        {
            "id": "good-1",
            "input": "SLA란 무엇인가요?",
            "expected_output": "SLA는 서비스 수준 합의입니다.",
            "context": ["SLA는 가용성, 지연, 오류율을 정의합니다."],
            "source_file": str(corpus_dir / "00_sla.md"),
            "topic": "SLA",
            "synthetic_input_quality": 0.9,
        },
        {
            "id": "good-2",
            "input": "Rewritten Input: 지연은 어떤 지표로 측정하나요?",
            "expected_output": "지연은 p50, p95, p99와 TTFE, E2E로 측정합니다.",
            "context": ["지연은 p50/p95/p99, TTFE, E2E로 측정합니다."],
            "source_file": str(corpus_dir / "00_sla.md"),
            "topic": "SLA",
            "synthetic_input_quality": 0.8,
        },
        {
            "id": "bad-1",
            "input": "Design corpus validator for Step1",
            "expected_output": "manifest와 subdirectories 규칙을 정의합니다.",
            "context": ["corpus rules and validator"],
            "source_file": str(corpus_dir / "AGENTS.md"),
            "topic": "corpus",
            "synthetic_input_quality": 0.95,
        },
    ]

    monkeypatch.setattr(
        "src.loop1_dataset.golden_builder.generate_synthetic_dataset",
        lambda **kwargs: candidate_items,
    )
    monkeypatch.setattr(
        "src.loop1_dataset.golden_builder.augment_expected_tools",
        lambda items, agent_module: [{**item, "expected_tools": [{"name": "search_web"}]} for item in items],
    )

    items = build_golden_dataset(num_goldens=2, skip_review=True)

    assert len(items) == 2
    assert all(item["approved"] is True for item in items)
    assert {item["input"] for item in items} == {
        "SLA란 무엇인가요?",
        "지연은 어떤 지표로 측정하나요?",
    }

    golden_path = tmp_data_dir / "golden" / "golden_dataset.json"
    with open(golden_path, encoding="utf-8") as f:
        saved = json.load(f)

    assert len(saved) == 2
    assert all(item["validation_status"] == "passed" for item in saved)
    assert all(item["expected_tools"] == [{"name": "search_web"}] for item in saved)


def test_build_golden_dataset_when_reviewed_csv_path_then_imports_and_validates(
    tmp_data_dir, monkeypatch
):
    from src.loop1_dataset.golden_builder import build_golden_dataset

    corpus_dir = tmp_data_dir / "corpus"
    corpus_dir.mkdir(exist_ok=True)

    monkeypatch.setattr(
        "src.loop1_dataset.golden_builder.get_settings",
        lambda: SimpleNamespace(data_dir=tmp_data_dir, local_corpus_dir=corpus_dir),
    )
    monkeypatch.setattr(
        "src.loop1_dataset.golden_builder.augment_expected_tools",
        lambda items, agent_module: items,
    )

    reviewed_csv_path = tmp_data_dir / "review" / "reviewed.csv"
    pd.DataFrame(
        [
            {
                "id": "approved-1",
                "input": "가용성은 어떻게 계산하나요?",
                "expected_output": "가용성은 1 - (장애 시간 / 총 시간)입니다.",
                "context": "가용성은 1 - (장애 시간 / 총 시간)으로 계산합니다.",
                "source_file": str(corpus_dir / "00_sla.md"),
                "synthetic_input_quality": 0.8,
                "approved": "True",
                "feedback": "",
                "reviewer": "tester",
                "topic": "SLA",
            },
            {
                "id": "rejected-1",
                "input": "Explain corpus purpose and validator",
                "expected_output": "corpus의 step1 규칙을 설명합니다.",
                "context": "corpus validation rules",
                "source_file": str(corpus_dir / "AGENTS.md"),
                "synthetic_input_quality": 0.9,
                "approved": "False",
                "feedback": "",
                "reviewer": "tester",
                "topic": "corpus",
            },
        ]
    ).to_csv(reviewed_csv_path, index=False, encoding="utf-8-sig")

    items = build_golden_dataset(
        num_goldens=1,
        reviewed_csv_path=reviewed_csv_path,
    )

    assert len(items) == 1
    assert items[0]["input"] == "가용성은 어떻게 계산하나요?"
    assert items[0]["approved"] is True
    assert items[0]["validation_status"] == "passed"
