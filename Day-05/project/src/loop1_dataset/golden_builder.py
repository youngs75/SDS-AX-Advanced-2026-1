"""Golden Dataset 빌더 - Loop 1 오케스트레이터 (Loop 1 - Step 4).

Loop 1의 핵심 책임은 LLM으로 Golden 후보를 생성하고,
자동 검증을 거쳐 실제 Golden Dataset을 확정하는 것입니다.

지원 모드:
    1. reviewed_csv_path 지정: 사람 검토가 끝난 CSV에서 Golden 재구성
    2. 기본/skip_review: LLM 후보 생성 → 자동 검증 → Golden 확정

Human Review CSV는 선택 기능입니다. Builder의 기본 동작은
실제로 사용할 Golden Dataset을 만들어 내는 것입니다.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.loop1_dataset.csv_exporter import export_to_review_csv
from src.loop1_dataset.csv_importer import import_reviewed_csv
from src.loop1_dataset.expected_tools_augmenter import augment_expected_tools
from src.loop1_dataset.feedback_augmenter import augment_with_feedback
from src.loop1_dataset.synthesizer import generate_synthetic_dataset
from src.loop1_dataset.validator import select_valid_goldens
from src.settings import get_settings

_CANDIDATE_MULTIPLIER = 3


def _save_json(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def _augment_feedback_if_needed(items: list[dict], golden_path: Path) -> list[dict]:
    items_with_feedback = [item for item in items if item.get("feedback")]
    if not items_with_feedback:
        return items

    print(f"[Loop1] {len(items_with_feedback)}개 항목 LLM 피드백 보강 중...")
    augmented_items = augment_with_feedback(items)
    _save_json(golden_path, augmented_items)
    return augmented_items


def _augment_expected_tools_if_needed(
    items: list[dict],
    *,
    agent_module: str,
    golden_path: Path,
) -> list[dict]:
    print(f"[Loop1] expected_tools 보강 중... (agent_module={agent_module})")
    augmented_items = augment_expected_tools(items, agent_module=agent_module)
    _save_json(golden_path, augmented_items)
    return augmented_items


def _finalize_golden_items(
    *,
    candidate_items: list[dict],
    target_count: int,
    golden_path: Path,
    review_csv_path: Path | None = None,
) -> list[dict]:
    final_items, validated_items = select_valid_goldens(
        candidate_items,
        target_count=target_count,
    )

    if review_csv_path is not None:
        review_json_path = review_csv_path.with_suffix(".auto_candidates.json")
        _save_json(review_json_path, validated_items)
        export_to_review_csv(review_json_path, review_csv_path)
        print(f"[Loop1] 자동 검증 후보 CSV 저장: {review_csv_path}")

    if len(final_items) < target_count:
        raise ValueError(
            f"자동 검증을 통과한 Golden 후보가 부족합니다: "
            f"requested={target_count}, validated={len(final_items)}"
        )

    for item in final_items:
        item["approved"] = True
        item["feedback"] = item.get("feedback", "")
        item["reviewer"] = item.get("reviewer", "auto")

    _save_json(golden_path, final_items)
    return final_items


def build_golden_dataset(
    *,
    corpus_dir: Path | None = None,
    num_goldens: int = 10,
    skip_review: bool = False,
    reviewed_csv_path: Path | None = None,
    agent_module: str = "src.my_agent",
) -> list[dict]:
    """Loop 1 전체를 오케스트레이션하여 Golden Dataset을 빌드합니다."""
    settings = get_settings()
    corpus_dir = corpus_dir or settings.local_corpus_dir
    data_dir = settings.data_dir

    synthetic_path = data_dir / "synthetic" / "synthetic_dataset.json"
    review_csv_path = data_dir / "review" / "review_dataset.csv"
    golden_path = data_dir / "golden" / "golden_dataset.json"

    if reviewed_csv_path is not None:
        if not reviewed_csv_path.exists():
            raise FileNotFoundError(f"리뷰된 CSV를 찾을 수 없습니다: {reviewed_csv_path}")

        print(f"[Loop1] 리뷰된 CSV 사용: {reviewed_csv_path}")
        imported_items = import_reviewed_csv(
            reviewed_csv_path,
            golden_path,
            only_approved=True,
        )
        golden_items = _finalize_golden_items(
            candidate_items=imported_items,
            target_count=min(num_goldens, len(imported_items)) if imported_items else num_goldens,
            golden_path=golden_path,
        )
        golden_items = _augment_feedback_if_needed(golden_items, golden_path)
        golden_items = _augment_expected_tools_if_needed(
            golden_items,
            agent_module=agent_module,
            golden_path=golden_path,
        )
        print(f"[Loop1] Golden Dataset 완성: {len(golden_items)}개 항목 → {golden_path}")
        return golden_items

    candidate_count = max(num_goldens, num_goldens * _CANDIDATE_MULTIPLIER)
    print(f"[Loop1] Golden 후보 {candidate_count}개 생성 중...")
    candidate_items = generate_synthetic_dataset(
        corpus_dir=corpus_dir,
        output_path=synthetic_path,
        num_goldens=candidate_count,
    )

    golden_items = _finalize_golden_items(
        candidate_items=candidate_items,
        target_count=num_goldens,
        golden_path=golden_path,
        review_csv_path=None if skip_review else review_csv_path,
    )
    golden_items = _augment_expected_tools_if_needed(
        golden_items,
        agent_module=agent_module,
        golden_path=golden_path,
    )

    print(
        f"[Loop1] 자동 검증 완료: {len(golden_items)}/{len(candidate_items)}개 항목을 Golden으로 확정"
    )
    print(f"[Loop1] Golden Dataset 완성: {len(golden_items)}개 항목 → {golden_path}")
    return golden_items
