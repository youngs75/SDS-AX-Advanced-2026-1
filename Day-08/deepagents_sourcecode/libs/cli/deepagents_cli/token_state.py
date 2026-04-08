"""그래프 상태에 `_context_tokens` 채널을 추가하는 미들웨어입니다.

필드가 체크포인트되었지만(세션 전반에 걸쳐 유지됨) 모델에 전달되지 않았습니다(`PrivateStateAttr`).  CLI는 모든 LLM 응답 및 컨텍스트
오프로드 후에 여기에 최신 전체 컨텍스트 토큰 수를 기록하고 스레드를 재개할 때 이를 다시 읽어 `/tokens` 및 상태 표시줄에 즉시 정확한 값이
표시되도록 합니다.
"""

from __future__ import annotations

from typing import Annotated, NotRequired

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    PrivateStateAttr,
)


class TokenTrackingState(AgentState):
    """지속형 컨텍스트 토큰 카운터를 사용하여 에이전트 상태를 확장합니다."""

    _context_tokens: Annotated[NotRequired[int], PrivateStateAttr]
    """모델의 마지막 `usage_metadata`에서 보고된 총 컨텍스트 토큰입니다."""


class TokenStateMiddleware(AgentMiddleware):
    """상태 스키마에 `_context_tokens`을 등록하는 스키마 전용 미들웨어입니다."""

    state_schema = TokenTrackingState
