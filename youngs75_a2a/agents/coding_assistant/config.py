"""Coding Assistant 에이전트 설정.

Generator-Verifier 분리 원칙 (RubricRewards 논문):
- generation_model: 코드 생성용 (요구사항만 보고 작업)
- verification_model: 검증용 (더 넓은 맥락에서 판단)
"""

from __future__ import annotations

import os

from pydantic import Field

from youngs75_a2a.core.config import BaseAgentConfig


class CodingConfig(BaseAgentConfig):
    """Coding Assistant 에이전트 설정."""

    generation_model: str = Field(
        default_factory=lambda: os.getenv("CODING_GEN_MODEL", "gpt-5.4"),
        description="코드 생성용 모델",
    )
    verification_model: str = Field(
        default_factory=lambda: os.getenv("CODING_VERIFY_MODEL", "gpt-5.4"),
        description="코드 검증용 모델",
    )

    # 허용 파일 확장자
    allowed_extensions: list[str] = Field(
        default=[".py", ".js", ".ts", ".json", ".yaml", ".yml", ".md", ".toml"],
    )
    # 최대 삭제 허용 줄 수
    max_delete_lines: int = Field(default=100)

    def _resolve_model_name(self, purpose: str) -> str:
        """목적별 모델 분기 — Generator/Verifier 분리."""
        if purpose == "generation":
            return self.generation_model
        if purpose == "verification":
            return self.verification_model
        return self.default_model
