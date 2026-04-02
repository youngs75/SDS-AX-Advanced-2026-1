"""OpenRouter 범용 클라이언트 모듈.

OpenAI SDK를 사용하여 OpenRouter API에 접근합니다.
base_url만 변경하면 OpenAI SDK의 모든 기능을 그대로 사용할 수 있습니다.

사용 예시:
    from src.llm.openrouter import get_openrouter_client
    client = get_openrouter_client()
    response = client.chat.completions.create(
        model="openai/gpt-4.1",
        messages=[{"role": "user", "content": "Hello!"}],
    )

핵심 개념:
    OpenRouter는 다양한 LLM 프로바이더(OpenAI, Anthropic, Google 등)를
    하나의 통합 API 엔드포인트로 제공합니다. OpenAI SDK와 100% 호환되므로
    base_url과 api_key만 바꾸면 됩니다.
"""

from __future__ import annotations

from openai import OpenAI

from src.settings import get_settings


def get_openrouter_client() -> OpenAI:
    """OpenRouter를 통한 범용 OpenAI SDK 클라이언트를 생성합니다.

    OpenAI SDK의 base_url을 OpenRouter로 변경하여,
    하나의 API 키로 다양한 모델(GPT-4, Claude, Gemini 등)에 접근할 수 있습니다.

    Returns:
        OpenAI: OpenRouter 엔드포인트가 설정된 OpenAI 클라이언트

    Note:
        이 클라이언트는 feedback_augmenter 등 범용 LLM 호출에 사용됩니다.
        DeepEval 전용 호출은 deepeval_model.py의 OpenRouterModel을 사용하세요.
    """
    settings = get_settings()
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=settings.openrouter_api_key,
    )
