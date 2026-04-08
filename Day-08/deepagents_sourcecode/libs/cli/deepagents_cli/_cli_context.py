"""CLI 모델 재정의를 위한 경량 런타임 컨텍스트 유형입니다.

`configurable_model`에서 추출되므로 핫 경로 모듈(`app`, `textual_adapter`)은 langchain 미들웨어 스택을 가져오지
않고도 `CLIContext`을(를) 가져올 수 있습니다.
"""

from __future__ import annotations

from typing import Any

from typing_extensions import TypedDict


class CLIContext(TypedDict, total=False):
    """`context=`을 통해 LangGraph 그래프로 전달된 런타임 컨텍스트입니다.

    `ConfigurableModelMiddleware`이 `request.runtime.context`에서 읽는 호출별 재정의를 수행합니다.

    """

    model: str | None
    """런타임 시 교체할 모델 사양(예: `'openai:gpt-4o'`)"""

    model_params: dict[str, Any]
    """병합할 호출 매개변수(예: `temperature`, `max_tokens`)
    `model_settings`에.
    """
