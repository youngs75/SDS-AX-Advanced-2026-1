"""검색 결과 랭킹 유틸리티."""

from __future__ import annotations

from typing import Any

import httpx


def late_fusion_with_rerank(
    results_lists: list[list[Any]],
    rerank_url: str | None = None,
    query: str = "",
    text_key: str = "caption_ko",
    k: int = 60,
    top_n: int = 5,
) -> list[dict]:
    """Late Fusion (RRF) + 선택적 Reranker로 결과를 병합합니다.

    Args:
        results_lists: 여러 검색 결과 리스트 (각각 ScoredPoint 또는 dict)
        rerank_url: TEI Reranker 서버 URL (None이면 RRF만 사용)
        query: 리랭킹용 원본 쿼리
        text_key: 리랭킹에 사용할 텍스트 필드명
        k: RRF 상수 (기본 60)
        top_n: 최종 반환 개수

    Returns:
        RRF (+ rerank) 정렬된 결과 리스트
    """
    # RRF 점수 계산
    doc_scores: dict[str, float] = {}
    doc_data: dict[str, dict] = {}

    for results in results_lists:
        for rank, item in enumerate(results):
            if hasattr(item, "id"):
                doc_id = str(item.id)
                payload = item.payload or {}
            else:
                doc_id = str(item.get("id", rank))
                payload = item

            rrf_score = 1.0 / (k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + rrf_score

            if doc_id not in doc_data:
                doc_data[doc_id] = {
                    "id": doc_id,
                    **payload,
                }

    # RRF 정렬
    sorted_ids = sorted(doc_scores, key=lambda x: doc_scores[x], reverse=True)
    fused = [
        {**doc_data[did], "rrf_score": doc_scores[did]}
        for did in sorted_ids[:top_n * 2]
    ]

    # Reranker 적용 (선택)
    if rerank_url and query and fused:
        try:
            texts = [doc.get(text_key, "") for doc in fused]
            client = httpx.Client(timeout=30.0)
            resp = client.post(
                f"{rerank_url.rstrip('/')}/rerank",
                json={"query": query, "texts": texts},
            )
            resp.raise_for_status()
            scores = resp.json()

            for i, score in enumerate(scores):
                if i < len(fused):
                    fused[i]["rerank_score"] = score if isinstance(score, float) else score.get("score", 0.0)

            fused.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        except Exception:
            pass  # Reranker 실패 시 RRF 결과 유지

    return fused[:top_n]
