"""LLM 클라이언트 선택 모듈.

프로젝트 기본값은 OpenRouter이지만, 필요 시 OpenAI direct 호출로도
전환할 수 있습니다. OpenAI SDK 하나로 두 경로를 모두 다룹니다.
"""

from __future__ import annotations

from pydantic import SecretStr
from openai import OpenAI
from langchain_openai import ChatOpenAI

from src.settings import Settings, get_settings


def get_llm_provider(settings: Settings | None = None) -> str:
    """현재 사용할 LLM provider를 반환합니다."""
    active_settings = settings or get_settings()
    provider = (active_settings.llm_provider or "openrouter").strip().lower()
    if provider not in {"openrouter", "openai"}:
        raise ValueError(f"Unsupported LLM provider: {provider}")
    return provider


def get_chat_model_name(settings: Settings | None = None) -> str:
    """현재 provider 기준 기본 chat model 이름을 반환합니다."""
    active_settings = settings or get_settings()
    provider = get_llm_provider(active_settings)
    if provider == "openai":
        return active_settings.openai_model_name
    return active_settings.openrouter_model_name


def get_llm_client(
    *,
    settings: Settings | None = None,
    provider: str | None = None,
    api_key: str | None = None,
) -> OpenAI:
    """현재 provider에 맞는 OpenAI SDK 클라이언트를 생성합니다."""
    active_settings = settings or get_settings()
    resolved_provider = (provider or get_llm_provider(active_settings)).strip().lower()

    if resolved_provider == "openai":
        resolved_api_key = api_key or active_settings.openai_api_key
        if not resolved_api_key:
            raise ValueError("OPENAI_API_KEY is not configured for provider=openai")
        return OpenAI(api_key=resolved_api_key)

    if resolved_provider == "openrouter":
        resolved_api_key = api_key or active_settings.openrouter_api_key
        if not resolved_api_key:
            raise ValueError("OPENROUTER_API_KEY is not configured for provider=openrouter")
        return OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=resolved_api_key,
        )

    raise ValueError(f"Unsupported LLM provider: {resolved_provider}")


def get_langchain_chat_model(
    *,
    settings: Settings | None = None,
    provider: str | None = None,
    model_name: str | None = None,
) -> ChatOpenAI:
    """LangChain/DeepAgents에서 사용할 ChatOpenAI 인스턴스를 생성합니다."""
    active_settings = settings or get_settings()
    resolved_provider = (provider or get_llm_provider(active_settings)).strip().lower()

    if resolved_provider == "openai":
        api_key = active_settings.openai_api_key
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not configured for provider=openai")
        return ChatOpenAI(
            api_key=SecretStr(api_key),
            model=model_name or active_settings.openai_model_name,
        )

    if resolved_provider == "openrouter":
        api_key = active_settings.openrouter_api_key
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not configured for provider=openrouter")
        return ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=SecretStr(api_key),
            model=model_name or active_settings.openrouter_model_name,
        )

    raise ValueError(f"Unsupported LLM provider: {resolved_provider}")


def get_openrouter_client() -> OpenAI:
    """하위 호환용 alias. 현재 provider 기준 클라이언트를 반환합니다."""
    return get_llm_client()
