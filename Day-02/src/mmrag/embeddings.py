"""Multi-Modal 임베딩 클라이언트 (TEI 호환)."""

from __future__ import annotations

import base64
from pathlib import Path

import httpx
import numpy as np


class HuggingfaceMMEmbedding:
    """TEI 서버 기반 멀티모달 임베딩 클라이언트.

    텍스트와 이미지를 동일한 벡터 공간에 임베딩합니다.
    Omni-Embed Nemotron 등 멀티모달 TEI 모델과 호환됩니다.
    """

    def __init__(
        self,
        text_url: str = "http://localhost:8001",
        image_url: str | None = None,
        timeout: float = 60.0,
        batch_size_text: int = 16,
        batch_size_image: int = 8,
    ):
        self.text_url = text_url.rstrip("/")
        self.image_url = (image_url or text_url).rstrip("/")
        self.timeout = timeout
        self.batch_size_text = batch_size_text
        self.batch_size_image = batch_size_image
        self.client = httpx.Client(timeout=timeout)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """텍스트 리스트를 임베딩합니다."""
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size_text):
            batch = texts[i : i + self.batch_size_text]
            resp = self.client.post(
                f"{self.text_url}/embed",
                json={"inputs": batch},
            )
            resp.raise_for_status()
            all_embeddings.extend(resp.json())
        return np.array(all_embeddings)

    def embed_images(self, image_paths: list[str | Path]) -> np.ndarray:
        """이미지 파일들을 base64 인코딩 후 임베딩합니다."""
        all_embeddings: list[list[float]] = []
        for i in range(0, len(image_paths), self.batch_size_image):
            batch_paths = image_paths[i : i + self.batch_size_image]
            inputs = []
            for p in batch_paths:
                with open(p, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                inputs.append(f"data:image/png;base64,{b64}")
            resp = self.client.post(
                f"{self.image_url}/embed",
                json={"inputs": inputs},
            )
            resp.raise_for_status()
            all_embeddings.extend(resp.json())
        return np.array(all_embeddings)

    def embed_query(self, text: str) -> list[float]:
        """단일 쿼리 텍스트를 임베딩합니다."""
        return self.embed_texts([text])[0].tolist()

    def health_check(self) -> dict:
        """TEI 서버 상태를 확인합니다."""
        result = {}
        for name, url in [("text", self.text_url), ("image", self.image_url)]:
            try:
                resp = self.client.get(f"{url}/info")
                result[name] = {"status": "ok", "info": resp.json()} if resp.status_code == 200 else {"status": f"error ({resp.status_code})"}
            except Exception as e:
                result[name] = {"status": f"unreachable ({e})"}
        return result
