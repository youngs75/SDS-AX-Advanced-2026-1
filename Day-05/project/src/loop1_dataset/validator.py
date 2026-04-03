"""Loop 1 자동 검증 모듈.

LLM이 생성한 후보 중 실제 Golden Dataset으로 올릴 항목만 선별합니다.
기본 전략은 다음과 같습니다.

1. 메타 문서/운영 지시문 기반 후보 제거
2. 질문 문구 정규화 (`Rewritten Input:` 접두어 제거)
3. source/context와의 주제 관련성 계산
4. 점수 기반으로 상위 후보만 최종 Golden으로 확정
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

_META_FILENAMES = {
    "agents.md",
    "readme.md",
    "changelog.md",
    "license",
    "license.md",
}

_META_TERMS = {
    "agents.md",
    "working in this directory",
    "testing requirements",
    "common patterns",
    "dependencies",
    "step1",
    "step4",
    "corpus",
    "validator",
    "manifest",
    "repo-root",
    "subdirs",
    "subdirectories",
}

_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "your",
    "will",
    "have",
    "about",
    "using",
    "used",
    "are",
    "how",
    "what",
    "when",
    "where",
    "which",
    "then",
    "than",
    "into",
    "service",
    "level",
    "agreement",
    "합니다",
    "입니다",
    "사용",
    "관련",
    "기반",
    "문서",
    "생성",
}


def _tokenize(text: str) -> list[str]:
    # 완전한 형태소 분석기는 아니지만,
    # "주제 단어가 실제로 겹치는지"를 빠르게 확인하기에는 이 정도가 충분합니다.
    return [
        token.lower()
        for token in re.findall(r"[A-Za-z0-9가-힣][A-Za-z0-9가-힣/_-]{1,}", text)
        if token.lower() not in _STOPWORDS
    ]


def _extract_keywords(context_text: str, source_file: str, topic: str) -> set[str]:
    # 키워드는 세 군데에서 모읍니다.
    # 1. topic 라벨
    # 2. 파일 이름
    # 3. 문서 제목/불릿/자주 등장한 단어
    #
    # 이렇게 여러 소스에서 단어를 모으면 문서가 짧아도
    # "이 질문이 진짜 이 문서를 설명하는가?"를 더 안정적으로 볼 수 있습니다.
    keywords = set()
    keywords.update(_tokenize(topic))

    basename = Path(source_file).stem
    keywords.update(_tokenize(re.sub(r"^\d+[_-]*", "", basename)))

    for line in context_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith("-"):
            keywords.update(_tokenize(stripped))

    for token, _ in Counter(_tokenize(context_text)).most_common(12):
        keywords.add(token)

    return {keyword for keyword in keywords if len(keyword) >= 2}


def normalize_candidate_item(item: dict) -> dict:
    """생성 후보의 문구를 최소 수준으로 정규화합니다."""
    normalized = dict(item)
    raw_input = str(normalized.get("input", "")).strip()
    lower_input = raw_input.lower()

    # 일부 모델은 질문 앞에 "Rewritten Input:" 같은 편집용 접두어를 붙입니다.
    # 사람이 읽는 Golden에는 불필요하므로 여기서 제거합니다.
    for prefix in ("rewritten input:", "rewritten input", "rewritten question:"):
        if lower_input.startswith(prefix):
            cleaned = raw_input[len(prefix) :].strip(" :\n\t")
            normalized["input"] = cleaned
            normalized["input_was_normalized"] = True
            break

    if "context" not in normalized or normalized["context"] is None:
        normalized["context"] = []

    return normalized


def validate_golden_candidate(item: dict) -> dict:
    """후보 항목 하나를 주제 적합성 기준으로 평가합니다."""
    normalized = normalize_candidate_item(item)
    source_file = str(normalized.get("source_file", ""))
    source_name = Path(source_file).name.lower()
    topic = str(normalized.get("topic", ""))
    input_text = str(normalized.get("input", ""))
    expected_output = str(normalized.get("expected_output", ""))
    context = normalized.get("context", [])
    context_text = "\n".join(context) if isinstance(context, list) else str(context)

    combined_text = f"{input_text}\n{expected_output}".lower()
    # 질문 + 답변에 실제 문서 주제 단어가 얼마나 들어 있는지 봅니다.
    keywords = _extract_keywords(context_text, source_file, topic)
    keyword_hits = sorted(keyword for keyword in keywords if keyword in combined_text)

    # "manifest", "validator", "step1"처럼 메타 문서에서 자주 나오는 단어가 보이면
    # 오프토픽 후보일 가능성이 높다고 판단합니다.
    meta_matches = sorted(term for term in _META_TERMS if term in combined_text)
    if source_name in _META_FILENAMES:
        meta_matches.append(source_name)

    # 답변 안의 단어가 실제 context에도 어느 정도 포함되는지 계산합니다.
    # 값이 너무 낮으면 문서 근거 없이 모델이 꾸며냈을 가능성이 큽니다.
    answer_tokens = _tokenize(expected_output)
    context_tokens = set(_tokenize(context_text))
    grounded_hits = sum(1 for token in answer_tokens if token in context_tokens)
    groundedness = grounded_hits / max(1, len(answer_tokens))

    # relevance는 "주제 관련성", groundedness는 "근거 일치성"입니다.
    # 둘을 합쳐 최종 validation_score를 만들고, 그 점수로 정렬합니다.
    relevance = len(keyword_hits) / max(1, min(6, len(keywords)))
    validation_score = round((relevance * 0.65) + (groundedness * 0.35), 4)

    reasons: list[str] = []
    if meta_matches:
        reasons.append(f"meta_terms={','.join(meta_matches[:4])}")
    if not keyword_hits:
        reasons.append("no_topic_overlap")
    if groundedness < 0.12:
        reasons.append("low_groundedness")

    # 통과 기준은 단순하지만 의도가 분명합니다.
    # 1. 메타 문서 흔적이 없어야 하고
    # 2. 주제 단어가 실제로 겹쳐야 하며
    # 3. 답변이 최소한의 근거 일치성을 가져야 합니다.
    passes = not meta_matches and bool(keyword_hits) and groundedness >= 0.12

    validated = dict(normalized)
    validated["topic_keywords"] = keyword_hits[:8]
    validated["groundedness_score"] = round(groundedness, 4)
    validated["topic_relevance_score"] = round(relevance, 4)
    validated["validation_score"] = validation_score
    validated["validation_status"] = "passed" if passes else "rejected"
    validated["validation_reasons"] = reasons
    return validated


def select_valid_goldens(items: list[dict], *, target_count: int) -> tuple[list[dict], list[dict]]:
    """후보 목록에서 유효한 Golden만 선별합니다."""
    validated_items = [validate_golden_candidate(item) for item in items]
    passed_items = [item for item in validated_items if item["validation_status"] == "passed"]

    # 점수가 높은 후보를 먼저 선택합니다.
    # 동점일 때는 DeepEval이 주는 synthetic_input_quality를 보조 기준으로 사용합니다.
    passed_items.sort(
        key=lambda item: (
            item.get("validation_score", 0.0),
            item.get("synthetic_input_quality", 0.0),
        ),
        reverse=True,
    )
    return passed_items[:target_count], validated_items
