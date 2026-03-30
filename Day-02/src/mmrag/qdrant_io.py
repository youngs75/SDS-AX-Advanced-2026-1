"""Qdrant Named Vectors 관련 유틸리티."""

from __future__ import annotations

import uuid

from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams


def create_named_vectors_collection(
    client: QdrantClient,
    collection_name: str,
    vector_configs: dict[str, int],
    distance: Distance = Distance.COSINE,
    on_disk: bool = True,
) -> bool:
    """Named Vectors 기반 Qdrant 컬렉션을 생성합니다.

    Args:
        client: Qdrant 클라이언트
        collection_name: 컬렉션 이름
        vector_configs: {벡터이름: 차원} 매핑 (예: {"text": 768, "image": 2048})
        distance: 거리 함수
        on_disk: 디스크 저장 여부

    Returns:
        True (신규 생성) / False (이미 존재)
    """
    if client.collection_exists(collection_name):
        return False

    vectors_config = {
        name: VectorParams(size=dim, distance=distance, on_disk=on_disk)
        for name, dim in vector_configs.items()
    }

    client.create_collection(
        collection_name=collection_name,
        vectors_config=vectors_config,
    )
    return True


def upsert_multimodal_documents(
    client: QdrantClient,
    collection_name: str,
    documents: list[dict],
    embeddings: dict[str, "numpy.ndarray"],
) -> int:
    """멀티모달 문서를 Named Vectors로 업서트합니다.

    Args:
        client: Qdrant 클라이언트
        collection_name: 컬렉션 이름
        documents: 문서 리스트 (각 문서는 id, caption_ko, metadata 등 포함)
        embeddings: {벡터이름: ndarray[N, dim]} 매핑

    Returns:
        업로드된 문서 수
    """
    points = []
    for i, doc in enumerate(documents):
        vectors = {}
        for vec_name, emb_array in embeddings.items():
            vectors[vec_name] = emb_array[i].tolist()

        payload = {
            "caption_ko": doc.get("caption_ko", ""),
            "caption_en": doc.get("caption_en", ""),
            "filename": doc.get("filename", ""),
            "topic": doc.get("metadata", {}).get("topic", ""),
            **doc.get("metadata", {}),
        }

        point_id = doc.get("id", str(uuid.uuid4()))
        if isinstance(point_id, int):
            points.append(
                models.PointStruct(id=point_id, vector=vectors, payload=payload)
            )
        else:
            points.append(
                models.PointStruct(id=str(point_id), vector=vectors, payload=payload)
            )

    client.upsert(collection_name=collection_name, points=points)
    return len(points)


def search_multimodal(
    client: QdrantClient,
    collection_name: str,
    query_vector: list[float],
    vector_name: str,
    limit: int = 5,
    query_filter: models.Filter | None = None,
) -> list[models.ScoredPoint]:
    """특정 Named Vector로 검색합니다."""
    return client.query_points(
        collection_name=collection_name,
        query=query_vector,
        using=vector_name,
        limit=limit,
        query_filter=query_filter,
    ).points


def search_cross_modal(
    client: QdrantClient,
    collection_name: str,
    query_vector: list[float],
    query_vector_name: str,
    target_vector_name: str,
    limit: int = 5,
) -> list[models.ScoredPoint]:
    """교차 모달 검색 (예: 텍스트 임베딩으로 이미지 벡터 검색).

    query_vector_name의 공간에서 생성된 벡터로
    target_vector_name 공간을 검색합니다.
    """
    return client.query_points(
        collection_name=collection_name,
        query=query_vector,
        using=target_vector_name,
        limit=limit,
    ).points
