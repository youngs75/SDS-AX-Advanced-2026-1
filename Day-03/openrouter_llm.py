# ----------------------------------------------------------------------------
# OpenAI / OpenRouter 모델 초기화 헬퍼
# ----------------------------------------------------------------------------
import os
from typing import Literal

from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def _resolve_api_context() -> tuple[str, str]:
    """선택된 API 키와 베이스 URL 정보를 반환합니다."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        msg = "OPENROUTER_API_KEY가 필요합니다."
        raise RuntimeError(msg)

    base_url = os.getenv("OPENROUTER_API_BASE") or "https://openrouter.ai/api/v1"

    return (api_key, base_url)


def create_openrouter_llm(
    model: str = "openai/gpt-4.1-mini",
    temperature: float = 0.3,
    max_tokens: int | None = None,
    **kwargs: object,
) -> ChatOpenAI:
    """OpenAI 호환 LLM 생성 헬퍼.

    Args:
        model: 모델 이름. OpenRouter에서는 provider/model 형식 사용 가능
               (예: openai/gpt-4o, anthropic/claude-3-sonnet, google/gemini-pro)
        temperature: 생성 온도 (0.0-2.0)
        max_tokens: 최대 생성 토큰 수
        **kwargs: ChatOpenAI에 전달할 추가 파라미터

    Returns:
        ChatOpenAI: 설정된 LLM 인스턴스
    """
    api_key, base_url = _resolve_api_context()

    openai_kwargs: dict = {
        "model": model,
        "api_key": api_key,
        "temperature": temperature,
        "max_retries": 3,
        "timeout": 60,
        **kwargs,
    }
    if max_tokens is not None:
        openai_kwargs["max_tokens"] = max_tokens
    if base_url:
        openai_kwargs["base_url"] = base_url
    return ChatOpenAI(**openai_kwargs)


def create_embedding_model(
    model: str = "openai/text-embedding-3-small",
    **kwargs: object,
) -> OpenAIEmbeddings:
    """OpenAI 호환 임베딩 모델 생성.

    Args:
        model: 임베딩 모델 이름. OpenRouter에서는 provider/model 형식 사용 가능
               (예: openai/text-embedding-3-small, openai/text-embedding-3-large)
        **kwargs: 추가 파라미터 (encoding_format 등은 model_kwargs로 전달됨)

    Returns:
        OpenAIEmbeddings: 설정된 임베딩 모델 인스턴스
    """
    api_key, base_url = _resolve_api_context()

    # 전달받은 kwargs에서 model_kwargs로 전달할 파라미터 분리
    # encoding_format, extra_headers 등은 model_kwargs로 전달
    model_kwargs: dict = {}
    embedding_kwargs: dict = {
        "model": model,
        "api_key": api_key,
        "show_progress_bar": True,
        "skip_empty": True,
    }

    # 전달받은 kwargs 처리
    for key, value in kwargs.items():
        # OpenRouter API 특정 파라미터는 model_kwargs로 전달
        if key in ("encoding_format"):
            model_kwargs[key] = value
        else:
            # 나머지는 OpenAIEmbeddings에 직접 전달
            embedding_kwargs[key] = value

    if base_url:
        embedding_kwargs["base_url"] = base_url

    # model_kwargs가 있으면 전달
    if model_kwargs:
        embedding_kwargs["model_kwargs"] = model_kwargs

    return OpenAIEmbeddings(**embedding_kwargs)


def create_embedding_model_direct(
    model: str = "qwen/qwen3-embedding-0.6b",
    encoding_format: Literal["float", "base64"] = "float",
    input_text: str | list[str] = "",
    **kwargs: object,
) -> list[float] | list[list[float]]:
    """OpenAI SDK를 직접 사용하여 임베딩 생성 (encoding_format 지원).

    LangChain의 OpenAIEmbeddings가 encoding_format을 지원하지 않을 때 사용.

    Args:
        model: 임베딩 모델 이름
        encoding_format: 인코딩 형식 ("float")
        input_text: 임베딩할 텍스트 (문자열 또는 문자열 리스트)
        **kwargs: 추가 파라미터

    Returns:
        임베딩 벡터 리스트 (단일 텍스트) 또는 리스트의 리스트 (여러 텍스트)
    """
    from openai import OpenAI  # noqa: PLC0415

    api_key, base_url = _resolve_api_context()

    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

    # input_text가 비어있으면 kwargs에서 가져오기
    if not input_text:
        input_text = kwargs.get("input", "")

    response = client.embeddings.create(
        model=model,
        input=input_text,
        encoding_format=encoding_format,
    )

    # 단일 텍스트인 경우 첫 번째 임베딩 반환
    if isinstance(input_text, str):
        return response.data[0].embedding
    # 여러 텍스트인 경우 모든 임베딩 반환
    return [item.embedding for item in response.data]


def get_available_model_types() -> dict[str, list[str]]:
    """OpenRouter에서 사용 가능한 모델 유형을 반환합니다.

    Returns:
        dict[str, list[str]]: 모델 유형별 모델 목록
    """
    return {
        "chat": [
            "openai/gpt-4.1",
            "openai/gpt-4.1-mini",
            "openai/gpt-4.1-nano",
            "anthropic/claude-sonnet-4-6",
            "anthropic/claude-haiku-4-5",
            "anthropic/claude-opus-4-6",
            "google/gemini-2.5-flash",
            "google/gemini-2.5-pro",
            "x-ai/grok-4",
            "moonshotai/kimi-k2",
            "deepseek/deepseek-r1",
        ],
        "embedding": [
            "openai/text-embedding-3-small",
            "openai/text-embedding-3-large",
            "google/gemini-embedding-001",
            "qwen/qwen3-embedding-0.6b",
            "qwen/qwen3-embedding-4b",
            "qwen/qwen3-embedding-8b",
        ],
    }
